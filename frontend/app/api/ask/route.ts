import { NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";
import OpenAI from "openai";

export const runtime = "nodejs";

const REFUSAL_MESSAGE =
  "Not enough information in the indexed AI/ML documents to answer confidently.";

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

function getOpenAIClient() {
  return new OpenAI({
    apiKey: env("OPENAI_API_KEY")
  });
}

function getGroqClient() {
  return new OpenAI({
    apiKey: env("GROQ_API_KEY"),
    baseURL: "https://api.groq.com/openai/v1"
  });
}

async function embedQuestion(question: string) {
  const provider = process.env.EMBEDDING_PROVIDER?.trim().toLowerCase() || "openai";
  if (provider !== "openai") {
    throw new Error("Only EMBEDDING_PROVIDER=openai is supported in this deployment.");
  }

  const response = await getOpenAIClient().embeddings.create({
    model: env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    input: question
  });

  return response.data[0].embedding;
}

async function retrieveDocuments(questionEmbedding: number[]) {
  const topK = numberEnv("TOP_K", 3);
  const threshold = numberEnv("SIMILARITY_THRESHOLD", 0.6);
  const supabase = getSupabaseClient();

  const { data, error } = await supabase.rpc("match_documents", {
    query_embedding: questionEmbedding,
    match_threshold: threshold,
    match_count: topK
  });

  if (error) {
    throw new Error(error.message);
  }

  const rows = ((data ?? []) as MatchedDocument[]).filter(
    (row) => Number(row.similarity) >= threshold
  );

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
- If the context is insufficient, respond exactly with:
  ${REFUSAL_MESSAGE}
- Include citations using the provided citation labels.
- Do not invent page numbers, source names, URLs, or facts.
- Keep answers student-friendly and concise.`;

  const response = await getGroqClient().chat.completions.create({
    model: env("GROQ_MODEL", "llama-3.1-8b-instant"),
    temperature: 0,
    messages: [
      { role: "system", content: systemPrompt },
      {
        role: "user",
        content: `Question: ${question}\n\nRetrieved context:\n${context}`
      }
    ]
  });

  return response.choices[0]?.message?.content?.trim() || REFUSAL_MESSAGE;
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
    const { chunks, topK, threshold } = await retrieveDocuments(embedding);
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
    const refusal = answer === REFUSAL_MESSAGE;

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

