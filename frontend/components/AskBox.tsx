"use client";

import { FormEvent, useState } from "react";
import { Loader2, Search, Send } from "lucide-react";
import { AnswerCard } from "./AnswerCard";

export type Citation = {
  chunk_id: string;
  source_title: string;
  page_start: number | null;
  page_end: number | null;
  source_url: string | null;
  similarity_score: number;
};

export type RetrievedChunk = Citation & {
  text: string;
};

export type AskTimings = {
  embedding_ms: number;
  retrieval_ms: number;
  generation_ms: number;
  total_ms: number;
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
  cache_hit?: boolean;
  embedding_cache_hit?: boolean;
  timings?: AskTimings;
};

const EXAMPLE_QUESTIONS = [
  "What is gradient boosting?",
  "Explain the bias-variance tradeoff.",
  "What is PCA?"
];

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
    setResponse(null);

    try {
      const result = await fetch("/api/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ question: trimmed })
      });

      if (!result.ok) {
        const errorBody = await result.json().catch(() => null);
        const errorText =
          errorBody?.error || errorBody?.detail || JSON.stringify(errorBody);
        throw new Error(
          errorText || `Request failed with status ${result.status}`
        );
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
      <div className="ask-header">
        <div>
          <span className="status-dot" aria-hidden="true" />
          <span>Grounded Q&A</span>
        </div>
        <strong>TOP_K 3</strong>
      </div>

      <form className="ask-form" onSubmit={submit}>
        <label htmlFor="question">Question</label>
        <div className="question-shell">
          <textarea
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask about regression, trees, boosting, PCA, evaluation metrics..."
            rows={4}
          />
          <div className="input-actions">
            <Search size={18} aria-hidden="true" />
            <button type="submit" disabled={loading || !question.trim()}>
              {loading ? (
                <Loader2 className="spin" size={18} aria-hidden="true" />
              ) : (
                <Send size={18} aria-hidden="true" />
              )}
              <span>{loading ? "Asking" : "Ask"}</span>
            </button>
          </div>
        </div>

        <div className="prompt-row" aria-label="Example questions">
          {EXAMPLE_QUESTIONS.map((prompt) => (
            <button
              className="prompt-chip"
              key={prompt}
              type="button"
              onClick={() => {
                setQuestion(prompt);
                setError(null);
              }}
            >
              {prompt}
            </button>
          ))}
        </div>
      </form>

      {loading ? (
        <div className="loading-panel" role="status" aria-live="polite">
          <Loader2 className="spin" size={18} aria-hidden="true" />
          <span>Retrieving sources and drafting answer</span>
        </div>
      ) : null}

      {error ? (
        <div className="error-banner">
          <strong>Request failed</strong>
          <span>{error}</span>
        </div>
      ) : null}
      {response ? <AnswerCard response={response} /> : null}
    </section>
  );
}
