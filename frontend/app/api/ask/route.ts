import { NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";

export const runtime = "nodejs";

const REFUSAL_MESSAGE =
  "Not enough information in the indexed AI/ML documents to answer confidently.";
const JINA_EMBEDDING_DIMENSIONS = 1024;

type MatchedDocument = {
  id: string;
  content: string;
  source_title: string;
  source_url: string | null;
  page_start: number | null;
  page_end: number | null;
  category: string | null;
  similarity: number;
};

type CandidateDocument = Omit<MatchedDocument, "similarity"> & {
  embedding?: string | number[] | null;
};

type Citation = {
  chunk_id: string;
  source_title: string;
  page_start: number | null;
  page_end: number | null;
  source_url: string | null;
  similarity_score: number;
};

type RetrievedChunk = Citation & {
  text: string;
};

type JinaEmbeddingResponse = {
  data: Array<{
    index: number;
    embedding: number[];
  }>;
};

type GroqChatResponse = {
  choices?: Array<{
    message?: {
      content?: string;
    };
  }>;
};

function env(name: string, fallback?: string) {
  const value = process.env[name]?.trim() || fallback;
  if (!value) {
    throw new Error(`${name} is required`);
  }
  return value;
}

function numberEnv(name: string, fallback: number) {
  const raw = process.env[name]?.trim();
  if (!raw) {
    return fallback;
  }

  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${name} must be a number`);
  }
  return parsed;
}

function getSupabaseClient() {
  return createClient(
    env("SUPABASE_URL"),
    env("SUPABASE_SERVICE_ROLE_KEY"),
    {
      auth: {
        persistSession: false,
        autoRefreshToken: false
      }
    }
  );
}

function normalizeJinaSimilarity(rawSimilarity: number) {
  const normalized = (rawSimilarity + 1) / 2;
  return Math.max(0, Math.min(1, normalized));
}

function parsePgVector(value: string | number[] | null | undefined) {
  if (Array.isArray(value)) {
    return value.map(Number);
  }
  if (!value) {
    return [];
  }
  return value
    .replace(/^\[|\]$/g, "")
    .split(",")
    .map((part) => Number(part.trim()))
    .filter(Number.isFinite);
}

function cosineSimilarity(left: number[], right: number[]) {
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;

  for (let index = 0; index < Math.min(left.length, right.length); index += 1) {
    dot += left[index] * right[index];
    leftNorm += left[index] * left[index];
    rightNorm += right[index] * right[index];
  }

  if (!leftNorm || !rightNorm) {
    return 0;
  }

  return dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm));
}

const STOP_WORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "between",
  "difference",
  "explain",
  "is",
  "the",
  "what"
]);

const TOPIC_PHRASES = [
  "linear regression",
  "logistic regression",
  "gradient boosting",
  "bagging",
  "boosting",
  "random forest",
  "decision tree",
  "impurity",
  "regularization",
  "ridge regression",
  "lasso regression",
  "principal component analysis",
  "pca",
  "k-means",
  "k means",
  "clustering",
  "cross-validation",
  "cross validation",
  "bias-variance",
  "bias variance",
  "overfitting",
  "precision",
  "recall",
  "supervised",
  "unsupervised",
  "feature engineering",
  "evaluation metrics",
  "svm",
  "support vector"
];

function searchTerms(question: string) {
  const normalized = question.toLowerCase().replace(/[^a-z0-9+\-\s]/g, " ");
  const terms = new Set<string>();

  TOPIC_PHRASES.forEach((phrase) => {
    if (normalized.includes(phrase)) {
      terms.add(phrase);
    }
  });

  normalized
    .split(/\s+/)
    .map((word) => word.trim())
    .filter((word) => word.length > 2 && !STOP_WORDS.has(word))
    .forEach((word) => terms.add(word));

  if (terms.has("pca")) {
    terms.add("principal component");
  }
  if (terms.has("k-means")) {
    terms.add("k means");
  }
  if (terms.has("impurity")) {
    terms.add("node impurity");
    terms.add("gini");
    terms.add("entropy");
    terms.add("information gain");
  }

  return [...terms].slice(0, 10);
}

function ilikeFilter(column: string, term: string) {
  const safeTerm = term.replace(/[%*,]/g, " ").replace(/\s+/g, " ").trim();
  return safeTerm ? `${column}.ilike.%${safeTerm}%` : null;
}

function keywordBoost(row: Omit<CandidateDocument, "embedding">, terms: string[]) {
  const haystack = `${row.source_title} ${row.category ?? ""} ${row.content}`.toLowerCase();
  let boost = 0;

  terms.forEach((term) => {
    const normalizedTerm = term.toLowerCase();
    const alternateTerm = normalizedTerm.replace(/-/g, " ");
    if (normalizedTerm.includes(" ") || normalizedTerm.includes("-")) {
      if (haystack.includes(normalizedTerm) || haystack.includes(alternateTerm)) {
        boost += 0.08;
      }
      return;
    }

    if (haystack.includes(normalizedTerm)) {
      boost += 0.02;
    }
  });

  return Math.min(boost, 0.16);
}

function looksLikeHtmlArtifact(text: string) {
  return /<span|<\/span|class=|&lt;span/i.test(text);
}

async function embedQuestion(question: string) {
  const provider = process.env.EMBEDDING_PROVIDER?.trim().toLowerCase() || "jina";
  if (provider !== "jina") {
    throw new Error("Only EMBEDDING_PROVIDER=jina is supported in this deployment.");
  }

  const response = await fetch("https://api.jina.ai/v1/embeddings", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env("JINA_API_KEY")}`
    },
    body: JSON.stringify({
      model: env("JINA_EMBEDDING_MODEL", "jina-embeddings-v3"),
      task: "retrieval.query",
      dimensions: JINA_EMBEDDING_DIMENSIONS,
      input: [question]
    })
  });

  if (!response.ok) {
    throw new Error(`Jina embedding request failed: ${await response.text()}`);
  }

  const payload = (await response.json()) as JinaEmbeddingResponse;
  const embedding = payload.data?.[0]?.embedding;
  if (!embedding?.length) {
    throw new Error("Jina embedding response did not include an embedding.");
  }
  return embedding;
}

async function retrieveLexicalCandidates(
  supabase: ReturnType<typeof getSupabaseClient>,
  question: string,
  questionEmbedding: number[]
) {
  const terms = searchTerms(question);
  if (!terms.length) {
    return [];
  }

  const phraseTerms = terms.filter((term) => term.includes(" ") || term.includes("-"));
  const candidateRows = new Map<string, CandidateDocument>();

  for (const termGroup of [phraseTerms, terms]) {
    const filters = termGroup.flatMap((term) =>
      [ilikeFilter("content", term), ilikeFilter("source_title", term)].filter(Boolean)
    );

    if (!filters.length) {
      continue;
    }

    const { data, error } = await supabase
      .from("documents")
      .select("id, content, source_title, source_url, page_start, page_end, category, embedding")
      .or(filters.join(","))
      .limit(60);

    if (error) {
      throw new Error(error.message);
    }

    ((data ?? []) as CandidateDocument[]).forEach((row) => {
      candidateRows.set(row.id, row);
    });
  }

  return [...candidateRows.values()]
    .map((row) => {
      const { embedding, ...document } = row;
      const rawSimilarity = cosineSimilarity(questionEmbedding, parsePgVector(embedding));
      return {
        ...document,
        similarity: Math.min(
          1,
          normalizeJinaSimilarity(rawSimilarity) + keywordBoost(document, terms)
        )
      };
    })
    .filter((row) => Number.isFinite(row.similarity) && !looksLikeHtmlArtifact(row.content));
}

async function retrieveDocuments(question: string, questionEmbedding: number[]) {
  const topK = numberEnv("TOP_K", 3);
  const threshold = numberEnv("SIMILARITY_THRESHOLD", 0.6);
  const supabase = getSupabaseClient();

  const { data, error } = await supabase.rpc("match_documents", {
    query_embedding: questionEmbedding,
    match_threshold: 0,
    match_count: topK
  });

  if (error) {
    throw new Error(error.message);
  }

  const vectorRows = ((data ?? []) as MatchedDocument[])
    .filter((row) => !looksLikeHtmlArtifact(row.content))
    .map((row) => ({
      ...row,
      similarity: normalizeJinaSimilarity(Number(row.similarity))
    }));
  const lexicalRows = await retrieveLexicalCandidates(
    supabase,
    question,
    questionEmbedding
  );
  const byId = new Map<string, MatchedDocument>();

  [...vectorRows, ...lexicalRows].forEach((row) => {
    const existing = byId.get(row.id);
    if (!existing || row.similarity > existing.similarity) {
      byId.set(row.id, row);
    }
  });

  const rows = [...byId.values()]
    .sort((left, right) => right.similarity - left.similarity)
    .filter((row) => Number(row.similarity) >= threshold);

  return {
    chunks: rows.slice(0, topK),
    topK,
    threshold
  };
}

function citationLabel(chunk: MatchedDocument, index: number) {
  const page =
    chunk.page_start && chunk.page_end && chunk.page_start !== chunk.page_end
      ? `pp. ${chunk.page_start}-${chunk.page_end}`
      : chunk.page_start
        ? `p. ${chunk.page_start}`
        : "page unavailable";
  return `[C${index}: ${chunk.source_title}, ${page}]`;
}

function formatContext(chunks: MatchedDocument[]) {
  const maxChars = numberEnv("MAX_CONTEXT_CHARS", 5000);
  const blocks: string[] = [];
  let remaining = maxChars;

  chunks.forEach((chunk, index) => {
    if (remaining <= 0) {
      return;
    }

    const label = citationLabel(chunk, index + 1);
    let text = chunk.content.trim();
    if (text.length > remaining) {
      text = `${text.slice(0, remaining).replace(/\s+\S*$/, "")}...`;
    }

    blocks.push(
      [
        label,
        `Source URL: ${chunk.source_url || "Unavailable"}`,
        `Similarity: ${Number(chunk.similarity).toFixed(4)}`,
        text
      ].join("\n")
    );
    remaining -= text.length;
  });

  return blocks.join("\n\n---\n\n");
}

async function generateAnswer(question: string, chunks: MatchedDocument[]) {
  const context = formatContext(chunks);
  const systemPrompt = `You are a strict AI/ML document assistant.

Rules:
- Answer only from the provided retrieved context.
- Do not use outside knowledge.
- The retrieved context has already passed the relevance threshold. If it contains formulas, algorithm steps, properties, or examples related to the question, synthesize a concise grounded answer from those details.
- Do not refuse merely because the context is technical or partial.
- If the context is insufficient, respond exactly with:
  ${REFUSAL_MESSAGE}
- Include citations using the provided citation labels.
- Do not invent page numbers, source names, URLs, or facts.
- Keep answers student-friendly and concise.`;

  const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env("GROQ_API_KEY")}`
    },
    body: JSON.stringify({
      model: env("GROQ_MODEL", "llama-3.1-8b-instant"),
      temperature: 0,
      messages: [
        { role: "system", content: systemPrompt },
        {
          role: "user",
          content: `Question: ${question}\n\nRetrieved context:\n${context}`
        }
      ]
    })
  });

  if (!response.ok) {
    throw new Error(`Groq request failed: ${await response.text()}`);
  }

  const payload = (await response.json()) as GroqChatResponse;
  return payload.choices?.[0]?.message?.content?.trim() || REFUSAL_MESSAGE;
}

function toCitation(chunk: MatchedDocument): Citation {
  return {
    chunk_id: chunk.id,
    source_title: chunk.source_title,
    page_start: chunk.page_start,
    page_end: chunk.page_end,
    source_url: chunk.source_url,
    similarity_score: Number(chunk.similarity)
  };
}

function toRetrievedChunk(chunk: MatchedDocument): RetrievedChunk {
  return {
    ...toCitation(chunk),
    text: chunk.content
  };
}

function isRefusalAnswer(answer: string) {
  return answer === REFUSAL_MESSAGE || answer.startsWith("Not enough information");
}

export async function POST(request: Request) {
  try {
    const body = await request.json().catch(() => null);
    const question = typeof body?.question === "string" ? body.question.trim() : "";

    if (!question) {
      return NextResponse.json(
        { error: "Question must not be empty." },
        { status: 400 }
      );
    }

    const embedding = await embedQuestion(question);
    const { chunks, topK, threshold } = await retrieveDocuments(question, embedding);
    const citations = chunks.map(toCitation);
    const retrievedChunks = chunks.map(toRetrievedChunk);
    const similarityScores = citations.map((citation) => citation.similarity_score);
    const bestScore = similarityScores[0] ?? null;

    if (!chunks.length) {
      return NextResponse.json({
        answer: REFUSAL_MESSAGE,
        citations,
        retrieved_chunks: retrievedChunks,
        similarity_scores: similarityScores,
        refusal: true,
        best_score: bestScore,
        top_k: topK,
        similarity_threshold: threshold
      });
    }

    const answer = await generateAnswer(question, chunks);
    const refusal = isRefusalAnswer(answer);

    return NextResponse.json({
      answer,
      citations,
      retrieved_chunks: retrievedChunks,
      similarity_scores: similarityScores,
      refusal,
      best_score: bestScore,
      top_k: topK,
      similarity_threshold: threshold
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Request failed";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
