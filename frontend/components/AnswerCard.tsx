import {
  AlertTriangle,
  CheckCircle2,
  Clock3,
  Database,
  Gauge
} from "lucide-react";
import type { AskResponse } from "./AskBox";
import { SourcesList } from "./SourcesList";

type AnswerCardProps = {
  response: AskResponse;
};

function cacheLabel(value: AskResponse["embedding_cache_hit"]) {
  if (value === "skipped") {
    return "skipped";
  }
  return value ? "hit" : "miss";
}

export function AnswerCard({ response }: AnswerCardProps) {
  const timings = response.timings;
  const cacheHit = Boolean(response.cache_hit);
  const embeddingCacheLabel = cacheHit
    ? "skipped"
    : cacheLabel(response.embedding_cache_hit);

  return (
    <div className="answer-layout">
      <article className={response.refusal ? "answer-card refusal" : "answer-card"}>
        <div className="answer-topline">
          <div className="answer-meta">
            {response.refusal ? (
              <AlertTriangle size={18} aria-hidden="true" />
            ) : (
              <CheckCircle2 size={18} aria-hidden="true" />
            )}
            <span>
              {response.refusal ? "Retrieval refused" : "Grounded answer"}
            </span>
          </div>

          <div className="status-badges" aria-label="Response metadata">
            {timings ? (
              <span className="status-badge">
                <Clock3 size={14} aria-hidden="true" />
                {timings.total_ms} ms
              </span>
            ) : null}
            <span className={cacheHit ? "status-badge good" : "status-badge"}>
              <Database size={14} aria-hidden="true" />
              {cacheHit ? "Query cache hit" : "Fresh answer"}
            </span>
          </div>
        </div>

        <div className="answer-body">
          <p>{response.answer}</p>
        </div>

        <div className="score-line">
          <span>Best score</span>
          <strong>
            {response.best_score === null
              ? "n/a"
              : response.best_score.toFixed(3)}
          </strong>
          <span>Threshold</span>
          <strong>{response.similarity_threshold.toFixed(2)}</strong>
        </div>

        {timings ? (
          <details className="timing-details">
            <summary>
              <Gauge size={15} aria-hidden="true" />
              Timing details
            </summary>
            <div className="timing-grid">
              <span>Query cache</span>
              <strong>{cacheHit ? "hit" : "miss"}</strong>
              <span>Embedding cache</span>
              <strong>{embeddingCacheLabel}</strong>
              <span>Embedding</span>
              <strong>{cacheHit ? "skipped" : `${timings.embedding_ms} ms`}</strong>
              <span>Retrieval</span>
              <strong>{cacheHit ? "skipped" : `${timings.retrieval_ms} ms`}</strong>
              <span>Generation</span>
              <strong>{cacheHit ? "skipped" : `${timings.generation_ms} ms`}</strong>
            </div>
          </details>
        ) : null}
      </article>

      <SourcesList
        citations={response.citations}
        retrievedChunks={response.retrieved_chunks}
      />
    </div>
  );
}
