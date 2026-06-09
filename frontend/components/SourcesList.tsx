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

function chunkPreview(text: string) {
  const compact = text.replace(/\s+/g, " ").trim();
  if (compact.length <= 760) {
    return compact;
  }
  return `${compact.slice(0, 760).replace(/\s+\S*$/, "")}...`;
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
                  {citation.source_url ? (
                    <a
                      className="source-title-link"
                      href={citation.source_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={(event) => event.stopPropagation()}
                    >
                      <strong>{citation.source_title}</strong>
                      <ExternalLink size={13} aria-hidden="true" />
                    </a>
                  ) : (
                    <strong>{citation.source_title}</strong>
                  )}
                  <span>{pageLabel(citation)}</span>
                </div>
                <b>
                  <span>Score</span>
                  {citation.similarity_score.toFixed(3)}
                </b>
              </summary>
              {chunk ? <p>{chunkPreview(chunk.text)}</p> : null}
              {citation.source_url ? (
                <a
                  className="source-open-link"
                  href={citation.source_url}
                  target="_blank"
                  rel="noopener noreferrer"
                >
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
