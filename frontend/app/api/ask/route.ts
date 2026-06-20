import { createHash } from "node:crypto";

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

type Timings = {
  embedding_ms: number;
  retrieval_ms: number;
  generation_ms: number;
  total_ms: number;
};

type EmbeddingCacheState = boolean | "skipped";
type CacheHitType = "none" | "exact" | "semantic";

type AskPayload = {
  answer: string;
  citations: Citation[];
  retrieved_chunks: RetrievedChunk[];
  similarity_scores: number[];
  refusal: boolean;
  best_score: number | null;
  top_k: number;
  similarity_threshold: number;
  cache_hit: boolean;
  cache_hit_type: CacheHitType;
  embedding_cache_hit: EmbeddingCacheState;
  semantic_cache_score?: number;
  matched_cached_question?: string;
  timings: Timings;
};

type CachedQueryRow = {
  normalized_question: string;
  answer: string;
  citations: unknown;
  retrieved_chunks: unknown;
  refusal: boolean;
};

type SemanticCachedQueryRow = CachedQueryRow & {
  id: string;
  similarity: number;
};

type CachedEmbeddingRow = {
  embedding: string | number[] | null;
};

type FallbackQueryCacheValue = {
  normalized_question: string;
  question_embedding: number[];
  answer: string;
  citations: Citation[];
  retrieved_chunks: RetrievedChunk[];
  refusal: boolean;
};

type FallbackEmbeddingCacheValue = {
  embedding: number[];
  normalized_question: string;
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

function booleanEnv(name: string, fallback: boolean) {
  const raw = process.env[name]?.trim().toLowerCase();
  if (!raw) {
    return fallback;
  }
  return ["1", "true", "yes", "on"].includes(raw);
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

function elapsedSince(start: number) {
  return Math.max(0, Math.round(performance.now() - start));
}

function normalizeQuestion(question: string) {
  return question
    .trim()
    .toLowerCase()
    .replace(/\s+/g, " ")
    .replace(/[?!.,;:]+$/g, "")
    .trim();
}

function sha256(value: string) {
  return createHash("sha256").update(value).digest("hex");
}

function jsonArray<T>(value: unknown): T[] {
  if (Array.isArray(value)) {
    return value as T[];
  }
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      return Array.isArray(parsed) ? (parsed as T[]) : [];
    } catch {
      return [];
    }
  }
  return [];
}

function cachedPayload(
  row: CachedQueryRow | FallbackQueryCacheValue,
  totalStart: number,
  options: {
    cacheHitType: Exclude<CacheHitType, "none">;
    embeddingCacheHit: EmbeddingCacheState;
    embeddingMs?: number;
    semanticCacheScore?: number;
    matchedCachedQuestion?: string;
  }
) {
  const citations = jsonArray<Citation>(row.citations);
  const retrievedChunks = jsonArray<RetrievedChunk>(row.retrieved_chunks);
  const similarityScores = citations.map((citation) => Number(citation.similarity_score));

  return {
    answer: row.answer,
    citations,
    retrieved_chunks: retrievedChunks,
    similarity_scores: similarityScores,
    refusal: row.refusal,
    best_score: similarityScores[0] ?? null,
    top_k: numberEnv("TOP_K", 3),
    similarity_threshold: numberEnv("SIMILARITY_THRESHOLD", 0.6),
    cache_hit: true,
    cache_hit_type: options.cacheHitType,
    embedding_cache_hit: options.embeddingCacheHit,
    ...(options.semanticCacheScore === undefined
      ? {}
      : { semantic_cache_score: options.semanticCacheScore }),
    ...(options.matchedCachedQuestion
      ? { matched_cached_question: options.matchedCachedQuestion }
      : {}),
    timings: {
      embedding_ms: options.embeddingMs ?? 0,
      retrieval_ms: 0,
      generation_ms: 0,
      total_ms: elapsedSince(totalStart)
    }
  } satisfies AskPayload;
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

const FALLBACK_CACHE_LIMIT = 200;
const fallbackQueryCache = new Map<string, FallbackQueryCacheValue>();
const fallbackEmbeddingCache = new Map<string, FallbackEmbeddingCacheValue>();

function rememberBounded<T>(cache: Map<string, T>, key: string, value: T) {
  if (cache.has(key)) {
    cache.delete(key);
  }
  cache.set(key, value);

  if (cache.size > FALLBACK_CACHE_LIMIT) {
    const oldestKey = cache.keys().next().value;
    if (oldestKey) {
      cache.delete(oldestKey);
    }
  }
}

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

async function readQueryCache(
  supabase: ReturnType<typeof getSupabaseClient>,
  questionHash: string,
  totalStart: number
) {
  if (!booleanEnv("CACHE_ENABLED", true)) {
    return null;
  }

  const { data, error } = await supabase
    .from("query_cache")
    .select("normalized_question, answer, citations, retrieved_chunks, refusal")
    .eq("question_hash", questionHash)
    .maybeSingle();

  if (error) {
    console.warn(`Query cache skipped: ${error.message}`);
  } else if (data) {
    return cachedPayload(data as CachedQueryRow, totalStart, {
      cacheHitType: "exact",
      embeddingCacheHit: "skipped"
    });
  }

  const fallback = fallbackQueryCache.get(questionHash);
  if (fallback) {
    return cachedPayload(fallback, totalStart, {
      cacheHitType: "exact",
      embeddingCacheHit: "skipped"
    });
  }

  return null;
}

async function saveQueryCache(
  supabase: ReturnType<typeof getSupabaseClient>,
  questionHash: string,
  normalizedQuestion: string,
  questionEmbedding: number[],
  payload: AskPayload
) {
  if (!booleanEnv("CACHE_ENABLED", true)) {
    return;
  }

  const { error } = await supabase
    .from("query_cache")
    .upsert(
      {
        question_hash: questionHash,
        normalized_question: normalizedQuestion,
        question_embedding: questionEmbedding,
        cache_source: "rag",
        answer: payload.answer,
        citations: payload.citations,
        retrieved_chunks: payload.retrieved_chunks,
        refusal: payload.refusal,
        updated_at: new Date().toISOString()
      },
      { onConflict: "question_hash" }
    );

  if (error) {
    console.warn(
      `Query cache save failed; using warm runtime fallback: ${error.message}`
    );
  }

  rememberBounded(fallbackQueryCache, questionHash, {
    normalized_question: normalizedQuestion,
    question_embedding: questionEmbedding,
    answer: payload.answer,
    citations: payload.citations,
    retrieved_chunks: payload.retrieved_chunks,
    refusal: payload.refusal
  });
}

async function readSemanticQueryCache(
  supabase: ReturnType<typeof getSupabaseClient>,
  questionEmbedding: number[]
) {
  if (
    !booleanEnv("CACHE_ENABLED", true) ||
    !booleanEnv("SEMANTIC_CACHE_ENABLED", true)
  ) {
    return null;
  }

  const threshold = numberEnv("SEMANTIC_CACHE_THRESHOLD", 0.9);
  const topK = Math.max(1, Math.floor(numberEnv("SEMANTIC_CACHE_TOP_K", 1)));
  const { data, error } = await supabase.rpc("match_query_cache", {
    query_embedding: questionEmbedding,
    match_threshold: threshold,
    match_count: topK
  });

  if (error) {
    console.warn(`Semantic query cache skipped: ${error.message}`);
  } else {
    const row = ((data ?? []) as SemanticCachedQueryRow[])[0];
    if (row && Number(row.similarity) >= threshold) {
      return {
        row,
        similarity: Number(row.similarity),
        normalizedQuestion: row.normalized_question
      };
    }
  }

  const fallbackMatches = [...fallbackQueryCache.values()]
    .map((row) => ({
      row,
      similarity: cosineSimilarity(questionEmbedding, row.question_embedding),
      normalizedQuestion: row.normalized_question
    }))
    .filter((match) => Number.isFinite(match.similarity))
    .sort((left, right) => right.similarity - left.similarity)
    .slice(0, topK);

  return fallbackMatches.find((match) => match.similarity >= threshold) ?? null;
}

async function getQuestionEmbedding(
  supabase: ReturnType<typeof getSupabaseClient>,
  normalizedQuestion: string,
  questionHash: string
) {
  if (booleanEnv("CACHE_ENABLED", true)) {
    const { data, error } = await supabase
      .from("embedding_cache")
      .select("embedding")
      .eq("question_hash", questionHash)
      .maybeSingle();

    if (error) {
      console.warn(`Embedding cache lookup skipped: ${error.message}`);
    } else if (data) {
      const embedding = parsePgVector((data as CachedEmbeddingRow).embedding);
      if (embedding.length === JINA_EMBEDDING_DIMENSIONS) {
        return {
          embedding,
          embeddingCacheHit: true
        };
      }

      console.warn(
        `Embedding cache row ignored: expected ${JINA_EMBEDDING_DIMENSIONS} dimensions, got ${embedding.length}`
      );
    }

    const fallback = fallbackEmbeddingCache.get(questionHash);
    if (fallback?.embedding.length === JINA_EMBEDDING_DIMENSIONS) {
      return {
        embedding: fallback.embedding,
        embeddingCacheHit: true
      };
    }
  }

  const embedding = await embedQuestion(normalizedQuestion);

  if (booleanEnv("CACHE_ENABLED", true)) {
    const { error } = await supabase
      .from("embedding_cache")
      .upsert(
        {
          question_hash: questionHash,
          normalized_question: normalizedQuestion,
          embedding
        },
        { onConflict: "question_hash" }
      );

    if (error) {
      console.warn(
        `Embedding cache save failed; using warm runtime fallback: ${error.message}`
      );
    }

    rememberBounded(fallbackEmbeddingCache, questionHash, {
      embedding,
      normalized_question: normalizedQuestion
    });
  }

  return {
    embedding,
    embeddingCacheHit: false
  };
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

async function retrieveDocuments(
  supabase: ReturnType<typeof getSupabaseClient>,
  question: string,
  questionEmbedding: number[]
) {
  const topK = numberEnv("TOP_K", 3);
  const threshold = numberEnv("SIMILARITY_THRESHOLD", 0.6);

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
  const maxChars = numberEnv("MAX_CONTEXT_CHARS", 3000);
  const perChunkLimit = Math.max(
    500,
    Math.floor(maxChars / Math.max(chunks.length, 1))
  );
  const blocks: string[] = [];
  let remaining = maxChars;

  chunks.forEach((chunk, index) => {
    if (remaining <= 0) {
      return;
    }

    const label = citationLabel(chunk, index + 1);
    const metadata = [
      label,
      `Source URL: ${chunk.source_url || "Unavailable"}`,
      `Similarity: ${Number(chunk.similarity).toFixed(4)}`
    ].join("\n");
    const textBudget = Math.min(perChunkLimit, remaining - metadata.length - 1);
    if (textBudget <= 0) {
      return;
    }

    let text = chunk.content.trim();
    if (text.length > textBudget) {
      text = `${text.slice(0, textBudget).replace(/\s+\S*$/, "")}...`;
    }

    const block = [metadata, text].join("\n");
    blocks.push(block);
    remaining -= block.length;
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
- Answer in 5-8 concise sentences unless the user asks for a detailed explanation.
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
      max_tokens: numberEnv("GROQ_MAX_TOKENS", 400),
      temperature: 0.2,
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
  const totalStart = performance.now();

  try {
    const body = await request.json().catch(() => null);
    const question = typeof body?.question === "string" ? body.question.trim() : "";

    if (!question) {
      return NextResponse.json(
        { error: "Question must not be empty." },
        { status: 400 }
      );
    }

    const normalizedQuestion =
      normalizeQuestion(question) || question.toLowerCase().trim();
    const questionHash = sha256(normalizedQuestion);
    const supabase = getSupabaseClient();
    const cachedAnswer = await readQueryCache(supabase, questionHash, totalStart);

    if (cachedAnswer) {
      return NextResponse.json(cachedAnswer);
    }

    const embeddingStart = performance.now();
    const { embedding, embeddingCacheHit } = await getQuestionEmbedding(
      supabase,
      normalizedQuestion,
      questionHash
    );
    const embeddingMs = elapsedSince(embeddingStart);

    const semanticMatch = await readSemanticQueryCache(supabase, embedding);
    if (semanticMatch) {
      const semanticPayload = cachedPayload(
        semanticMatch.row,
        totalStart,
        {
          cacheHitType: "semantic",
          embeddingCacheHit,
          embeddingMs,
          semanticCacheScore: semanticMatch.similarity,
          matchedCachedQuestion: semanticMatch.normalizedQuestion
        }
      );
      semanticPayload.timings.total_ms = elapsedSince(totalStart);
      return NextResponse.json(semanticPayload);
    }

    const retrievalStart = performance.now();
    const { chunks, topK, threshold } = await retrieveDocuments(
      supabase,
      question,
      embedding
    );
    const retrievalMs = elapsedSince(retrievalStart);
    const citations = chunks.map(toCitation);
    const retrievedChunks = chunks.map(toRetrievedChunk);
    const similarityScores = citations.map((citation) => citation.similarity_score);
    const bestScore = similarityScores[0] ?? null;

    if (!chunks.length) {
      const refusalPayload = {
        answer: REFUSAL_MESSAGE,
        citations,
        retrieved_chunks: retrievedChunks,
        similarity_scores: similarityScores,
        refusal: true,
        best_score: bestScore,
        top_k: topK,
        similarity_threshold: threshold,
        cache_hit: false,
        cache_hit_type: "none",
        embedding_cache_hit: embeddingCacheHit,
        timings: {
          embedding_ms: embeddingMs,
          retrieval_ms: retrievalMs,
          generation_ms: 0,
          total_ms: elapsedSince(totalStart)
        }
      } satisfies AskPayload;

      await saveQueryCache(
        supabase,
        questionHash,
        normalizedQuestion,
        embedding,
        refusalPayload
      );

      refusalPayload.timings.total_ms = elapsedSince(totalStart);
      return NextResponse.json(refusalPayload);
    }

    const generationStart = performance.now();
    const answer = await generateAnswer(question, chunks);
    const generationMs = elapsedSince(generationStart);
    const refusal = isRefusalAnswer(answer);

    const payload = {
      answer,
      citations,
      retrieved_chunks: retrievedChunks,
      similarity_scores: similarityScores,
      refusal,
      best_score: bestScore,
      top_k: topK,
      similarity_threshold: threshold,
      cache_hit: false,
      cache_hit_type: "none",
      embedding_cache_hit: embeddingCacheHit,
      timings: {
        embedding_ms: embeddingMs,
        retrieval_ms: retrievalMs,
        generation_ms: generationMs,
        total_ms: elapsedSince(totalStart)
      }
    } satisfies AskPayload;

    await saveQueryCache(
      supabase,
      questionHash,
      normalizedQuestion,
      embedding,
      payload
    );

    payload.timings.total_ms = elapsedSince(totalStart);
    return NextResponse.json(payload);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Request failed";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
