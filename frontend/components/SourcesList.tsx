import { ExternalLink } from "lucide-react";
import type { Citation, RetrievedChunk } from "./AskBox";

type SourcesListProps = {
  citations: Citation[];
  retrievedChunks: RetrievedChunk[];
};

function pageLabel(citation: Citation) {
  if (citation.page_start && citation.page_end) {
    return citation.page_start === citation.page_end
      ? `p. ${citation.page_start}`
      : `pp. ${citation.page_start}-${citation.page_end}`;
  }
  return "page unavailable";
}

export function SourcesList({ citations, retrievedChunks }: SourcesListProps) {
  if (!citations.length && !retrievedChunks.length) {
    return null;
  }

  return (
    <aside className="sources-panel" aria-label="Retrieved sources">
      <div className="sources-header">
        <h2>Sources</h2>
        <span>{citations.length}</span>
      </div>

      <div className="source-list">
        {citations.map((citation, index) => {
          const chunk = retrievedChunks[index];
          return (
            <details key={citation.chunk_id} className="source-item">
              <summary>
                <div>
                  <strong>{citation.source_title}</strong>
                  <span>{pageLabel(citation)}</span>
                </div>
                <b>{citation.similarity_score.toFixed(3)}</b>
              </summary>
              {chunk ? <p>{chunk.text}</p> : null}
              {citation.source_url ? (
                <a href={citation.source_url} target="_blank" rel="noreferrer">
                  <ExternalLink size={14} aria-hidden="true" />
                  Source
                </a>
              ) : null}
            </details>
          );
        })}
      </div>
    </aside>
  );
}
