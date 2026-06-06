"use client";

import { FormEvent, useState } from "react";
import { Loader2, Send } from "lucide-react";
import { AnswerCard } from "./AnswerCard";

export type Citation = {
  chunk_id: string;
  source_title: string;
  file_name: string;
  page_start: number | null;
  page_end: number | null;
  source_url: string | null;
  category: string | null;
  similarity_score: number;
};

export type RetrievedChunk = Citation & {
  text: string;
};

export type AskResponse = {
  answer: string;
  citations: Citation[];
  retrieved_chunks: RetrievedChunk[];
  similarity_scores: number[];
  refusal: boolean;
  best_score: number | null;
  top_k: number;
  similarity_threshold: number;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export function AskBox() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await fetch(`${API_BASE_URL}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ question: trimmed })
      });

      if (!result.ok) {
        throw new Error(`Request failed with status ${result.status}`);
      }

      const data = (await result.json()) as AskResponse;
      setResponse(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="ask-surface" aria-label="Ask the RAG assistant">
      <form className="ask-form" onSubmit={submit}>
        <label htmlFor="question">Question</label>
        <div className="input-row">
          <textarea
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="What is overfitting?"
            rows={4}
          />
          <button type="submit" disabled={loading || !question.trim()}>
            {loading ? (
              <Loader2 className="spin" size={18} aria-hidden="true" />
            ) : (
              <Send size={18} aria-hidden="true" />
            )}
            <span>{loading ? "Asking" : "Ask"}</span>
          </button>
        </div>
      </form>

      {error ? <div className="error-banner">{error}</div> : null}
      {response ? <AnswerCard response={response} /> : null}
    </section>
  );
}
