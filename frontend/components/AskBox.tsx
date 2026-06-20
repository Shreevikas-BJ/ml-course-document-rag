"use client";

import { FormEvent, useEffect, useState } from "react";
import { History, Loader2, MessageSquare, Search, Send } from "lucide-react";
import { AnswerCard } from "./AnswerCard";
import { SessionHistory, type SessionHistoryItem } from "./SessionHistory";

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

export type CacheHitType = "none" | "exact" | "semantic";

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
  cache_hit_type?: CacheHitType;
  embedding_cache_hit?: boolean | "skipped";
  semantic_cache_score?: number;
  matched_cached_question?: string;
  timings?: AskTimings;
};

const EXAMPLE_QUESTIONS = [
  "What is gradient boosting?",
  "Explain the bias-variance tradeoff.",
  "What is PCA?"
];

const SESSION_HISTORY_KEY = "rag_session_history";
const MAX_SESSION_HISTORY = 50;

export function AskBox() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<"ask" | "history">("ask");
  const [history, setHistory] = useState<SessionHistoryItem[]>([]);

  useEffect(() => {
    try {
      const stored = window.sessionStorage.getItem(SESSION_HISTORY_KEY);
      const parsed = stored ? JSON.parse(stored) : [];
      if (Array.isArray(parsed)) {
        setHistory(parsed.slice(0, MAX_SESSION_HISTORY));
      }
    } catch {
      window.sessionStorage.removeItem(SESSION_HISTORY_KEY);
    }
  }, []);

  function addHistory(questionText: string, data: AskResponse) {
    const item: SessionHistoryItem = {
      ...data,
      id: window.crypto.randomUUID(),
      question: questionText,
      created_at: new Date().toISOString()
    };
    const next = [item, ...history].slice(0, MAX_SESSION_HISTORY);
    window.sessionStorage.setItem(SESSION_HISTORY_KEY, JSON.stringify(next));
    setHistory(next);
  }

  function viewHistoryItem(item: SessionHistoryItem) {
    const { id: _id, question: savedQuestion, created_at: _createdAt, ...savedResponse } = item;
    setQuestion(savedQuestion);
    setResponse(savedResponse);
    setError(null);
    setActiveTab("ask");
  }

  function clearHistory() {
    window.sessionStorage.removeItem(SESSION_HISTORY_KEY);
    setHistory([]);
  }

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
      addHistory(trimmed, data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="ask-surface" aria-label="Ask the RAG assistant">
      <div className="workspace-tabs" role="tablist" aria-label="RAG workspace">
        <button
          type="button"
          role="tab"
          aria-controls="ask-tab-panel"
          aria-selected={activeTab === "ask"}
          className={activeTab === "ask" ? "workspace-tab active" : "workspace-tab"}
          onClick={() => setActiveTab("ask")}
        >
          <MessageSquare size={16} aria-hidden="true" />
          Ask
        </button>
        <button
          type="button"
          role="tab"
          aria-controls="history-tab-panel"
          aria-selected={activeTab === "history"}
          className={activeTab === "history" ? "workspace-tab active" : "workspace-tab"}
          onClick={() => setActiveTab("history")}
        >
          <History size={16} aria-hidden="true" />
          History
          {history.length ? <span>{history.length}</span> : null}
        </button>
      </div>

      {activeTab === "ask" ? (
        <div className="ask-panel" id="ask-tab-panel" role="tabpanel">
          <div className="ask-header">
            <div>
              <span className="status-dot" aria-hidden="true" />
              <span>Grounded Q&A</span>
            </div>
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
        </div>
      ) : (
        <SessionHistory
          items={history}
          onClear={clearHistory}
          onSelect={viewHistoryItem}
        />
      )}
    </section>
  );
}
