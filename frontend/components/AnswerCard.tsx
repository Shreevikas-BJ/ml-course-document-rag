import { AlertTriangle, CheckCircle2 } from "lucide-react";
import type { AskResponse } from "./AskBox";
import { SourcesList } from "./SourcesList";

type AnswerCardProps = {
  response: AskResponse;
};

export function AnswerCard({ response }: AnswerCardProps) {
  return (
    <div className="answer-layout">
      <article className={response.refusal ? "answer-card refusal" : "answer-card"}>
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
        <p>{response.answer}</p>
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
      </article>

      <SourcesList
        citations={response.citations}
        retrievedChunks={response.retrieved_chunks}
      />
    </div>
  );
}
