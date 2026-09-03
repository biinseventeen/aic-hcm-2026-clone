import { useEffect, useState } from "react";
import "./history.css";
import { HistoryService } from "../services/historyService";
import { HistoryRecord, RetrievalMode } from "../services/types";

export type HistoryItem = HistoryRecord;

interface HistoryProps {
  mode?: RetrievalMode;
  title?: string;
  items?: HistoryRecord[];
  isLoading?: boolean;
  onSelectItem?: (item: HistoryRecord) => void;
  onRefresh?: () => void;
}

function History({
  mode = "kis",
  title,
  items,
  isLoading = false,
  onSelectItem,
}: HistoryProps) {
  const [internalItems, setInternalItems] = useState<HistoryRecord[]>([]);
  const [loadingInternal, setLoadingInternal] = useState(false);

  useEffect(() => {
    if (items) {
      setInternalItems(items);
    } else {
      // Tự động tải từ HistoryService nếu không truyền props từ ngoài
      setLoadingInternal(true);
      HistoryService.getHistory(mode)
        .then((records) => setInternalItems(records))
        .catch(() => setInternalItems([]))
        .finally(() => setLoadingInternal(false));
    }
  }, [items, mode]);

  const getDefaultTitle = () => {
    switch (mode) {
      case "qna":
        return "QNA_HISTORY";
      case "trake":
        return "TRAKE_HISTORY";
      default:
        return "KIS_HISTORY";
    }
  };

  const displayTitle = title || getDefaultTitle();
  const displayLoading = isLoading || loadingInternal;

  return (
    <aside className="history-panel" data-component="History">
      <header className="history-panel__header_01">
        <h2 className="history-panel__title_01">{displayTitle}</h2>
      </header>

      <div className="history-panel__list_01">
        {displayLoading ? (
          <div className="history-panel__loading_01">
            <span className="history-panel__pulse_01">LOADING_HISTORY...</span>
          </div>
        ) : internalItems.length === 0 ? (
          <div className="history-panel__empty_01">
            <span className="history-panel__empty-title_01">NO_HISTORY_LOGGED</span>
            <p className="history-panel__empty-desc_01">Queries will appear here once executed.</p>
          </div>
        ) : (
          internalItems.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`history-panel__item_01 ${
                item.isActive ? "history-panel__item_01--active" : ""
              }`}
              onClick={() => onSelectItem?.(item)}
            >
              <span className="history-panel__time_01">{item.timestamp}</span>
              <span className="history-panel__query_01">{item.query}</span>
              <span className="history-panel__hits_01">HITS: {item.hits}</span>
            </button>
          ))
        )}
      </div>
    </aside>
  );
}

export default History;



