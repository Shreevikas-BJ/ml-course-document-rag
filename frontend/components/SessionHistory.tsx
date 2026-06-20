import { Clock3, History, Trash2 } from "lucide-react";
import type { AskResponse } from "./AskBox";

export type SessionHistoryItem = AskResponse & {
  id: string;
  question: string;
  created_at: string;
};

type SessionHistoryProps = {
  items: SessionHistoryItem[];
  onClear: () => void;
  onSelect: (item: SessionHistoryItem) => void;
};

function cacheLabel(item: SessionHistoryItem) {
  if (!item.cache_hit) {
    return "Fresh";
  }
  return item.cache_hit_type === "semantic" ? "Semantic cache" : "Exact cache";
}

function timeLabel(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "This session";
  }
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit"
  }).format(date);
}

export function SessionHistory({
  items,
  onClear,
  onSelect
}: SessionHistoryProps) {
  return (
    <section
      className="history-panel"
      id="history-tab-panel"
      role="tabpanel"
      aria-label="Session history"
    >
      <div className="history-header">
        <div>
          <h2>Session history</h2>
          <span>{items.length} saved</span>
        </div>
        {items.length ? (
          <button
            className="clear-history-button"
            type="button"
            onClick={onClear}
            aria-label="Clear session history"
            title="Clear session history"
          >
            <Trash2 size={16} aria-hidden="true" />
          </button>
        ) : null}
      </div>

      {items.length ? (
        <div className="history-list">
          {items.map((item) => (
            <button
              className="history-item"
              key={item.id}
              type="button"
              onClick={() => onSelect(item)}
            >
              <strong>{item.question}</strong>
              <span className="history-meta">
                <span>
                  <Clock3 size={13} aria-hidden="true" />
                  {timeLabel(item.created_at)}
                </span>
                <span className={item.cache_hit ? "history-cache hit" : "history-cache"}>
                  {cacheLabel(item)}
                </span>
                <span>{item.citations.length} sources</span>
              </span>
            </button>
          ))}
        </div>
      ) : (
        <div className="history-empty">
          <History size={22} aria-hidden="true" />
          <strong>No questions yet</strong>
        </div>
      )}
    </section>
  );
}
