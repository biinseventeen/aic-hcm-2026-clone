import "./detect.css";
import { downloadCSV } from "../utils/exportUtils";
import { DetectionRecord, DetectTagItem } from "../services/types";

export type DetectTag = DetectTagItem;
export type DetectItem = DetectionRecord;

interface DetectProps {
  items?: DetectionRecord[];
  isLoading?: boolean;
  onExport?: () => void;
  onSelectItem?: (item: DetectionRecord) => void;
}

function Detect({
  items,
  isLoading = false,
  onExport,
  onSelectItem,
}: DetectProps) {
  const internalItems = items ?? [];

  const handleDefaultExport = () => {
    if (onExport) {
      onExport();
      return;
    }
    const timestampStr = new Date().toISOString().slice(11, 19).replace(/:/g, "");
    const rows = internalItems.map((item) => [
      item.videoSource,
      item.frameId,
      item.confidence,
      item.tags.map((t) => t.label).join("|"),
    ]);
    downloadCSV(`detections_export_${timestampStr}.csv`, rows);
  };

  const displayLoading = isLoading;

  return (
    <aside className="detect-panel" data-component="Detect">
      {/* 1. Header Bar */}
      <header className="detect-panel__header_01">
        <h2 className="detect-panel__title_01">SCAN DETECT</h2>
        <button
          type="button"
          className="detect-panel__info-btn_01"
          aria-label="Detection Info"
          title="Thông tin phân tích đối tượng Faster R-CNN"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="16" x2="12" y2="12" />
            <line x1="12" y1="8" x2="12.01" y2="8" strokeWidth="3" />
          </svg>
        </button>
      </header>

      {/* 2. Action Bar */}
      <div className="detect-panel__action-bar_01">
        <button
          type="button"
          className="detect-panel__export-btn_01"
          onClick={handleDefaultExport}
          disabled={internalItems.length === 0}
        >
          EXPORT FILE (.CSV)
        </button>
      </div>

      {/* 3. Detection Card List */}
      <div className="detect-panel__list_01">
        {displayLoading ? (
          <div className="detect-panel__loading_01">
            <span className="detect-panel__pulse_01">SCANNING_OBJECTS...</span>
          </div>
        ) : internalItems.length === 0 ? (
          <div className="detect-panel__empty_01">
            <span className="detect-panel__empty-title_01">NO_DETECTIONS_RECORDED</span>
            <p className="detect-panel__empty-desc_01">Detected objects in current frame will appear here.</p>
          </div>
        ) : (
          internalItems.map((item) => (
            <article
              key={item.id}
              className={`detect-panel__card_01 ${
                item.isSelected ? "detect-panel__card_01--selected" : ""
              }`}
              onClick={() => onSelectItem?.(item)}
            >
              {/* Thumbnail Box */}
              <div className="detect-panel__image-box_01">
                <span className="detect-panel__source-badge_01">{item.videoSource}</span>

                <span
                  className={`detect-panel__conf-badge_01 ${
                    item.isAlert ? "detect-panel__conf-badge_01--alert" : ""
                  }`}
                >
                  {item.isAlert && <span className="detect-panel__conf-dot_01">● </span>}
                  {item.confidence}
                </span>

                {item.imageSrc ? (
                  <img
                    src={item.imageSrc}
                    alt={`Frame ${item.frameId}`}
                    className="detect-panel__image_01"
                  />
                ) : (
                  <div className="detect-panel__placeholder_01" />
                )}
              </div>

              {/* Footer Metadata */}
              <div className="detect-panel__footer_01">
                <div className="detect-panel__meta-row_01">
                  <span className="detect-panel__frame-label_01">
                    FRAME: <strong>{item.frameId}</strong>
                  </span>
                  <span className="detect-panel__time_01">{item.timestamp}</span>
                </div>

                {/* Tags list */}
                <div className="detect-panel__tags-row_01">
                  {item.tags.map((tag, idx) => (
                    <span
                      key={idx}
                      className={`detect-panel__tag_01 ${
                        tag.variant === "solid"
                          ? "detect-panel__tag_01--solid"
                          : "detect-panel__tag_01--text"
                      }`}
                    >
                      {tag.label}
                    </span>
                  ))}
                </div>
              </div>
            </article>
          ))
        )}
      </div>
    </aside>
  );
}

export default Detect;

