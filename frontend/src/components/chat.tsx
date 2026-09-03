import { useState, useRef, useEffect, useCallback } from "react";
import "./chat.css";
import { downloadCSV, downloadFrameImage } from "../utils/exportUtils";
import { ChatService } from "../services/chatService";
import {
  ChatMessage,
  RetrievalMode,
  SolveResponse,
  SystemAlert,
} from "../services/types";

interface ChatProps {
  mode?: RetrievalMode;
  onQueryExecuted?: (
    query: string,
    response: SolveResponse,
    files?: File[],
  ) => void;
}

function Chat({ mode = "kis", onQueryExecuted }: ChatProps) {
  const [inputValue, setInputValue] = useState("");
  const [attachedFiles, setAttachedFiles] = useState<File[]>([]);
  const [isFilesModalOpen, setIsFilesModalOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isThinking, setIsThinking] = useState(false);
  const [systemAlert, setSystemAlert] = useState<SystemAlert | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Kiểm tra trạng thái hệ thống thực tế
  const performSystemCheck = useCallback(async () => {
    const alert = await ChatService.checkSystemStatus();
    setSystemAlert(alert);
  }, []);

  useEffect(() => {
    // Kiểm tra ngay khi mount
    performSystemCheck();

    // Lắng nghe sự kiện mạng của trình duyệt
    const handleOnline = () => performSystemCheck();
    const handleOffline = () => {
      setSystemAlert({
        type: "network_offline",
        title: "NETWORK_OFFLINE_",
        description:
          "No internet/network connection detected. Browser is currently offline.",
        level: "error",
      });
    };

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, [performSystemCheck]);

  const getModeTitle = () => {
    switch (mode) {
      case "qna":
        return "CHAT AGENT [QNA_MODE]";
      case "trake":
        return "CHAT AGENT [TRAKE_MODE]";
      default:
        return "CHAT AGENT";
    }
  };

  const getPlaceholder = () => {
    switch (mode) {
      case "qna":
        return "Ask question about visual scene or entity...";
      case "trake":
        return "Enter sequence query (e.g., A THEN B THEN C)...";
      default:
        return "Enter retrieval query or command...";
    }
  };

  // 1. Xử lý đính kèm nhiều File / Video
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const newFiles = Array.from(e.target.files);
      setAttachedFiles((prev) => [...prev, ...newFiles]);
    }
  };

  const handleTriggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const handleRemoveSingleFile = (idx: number) => {
    setAttachedFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleClearAllFiles = () => {
    setAttachedFiles([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  // 2. Gửi truy vấn tới API Backend
  const handleSendQuery = async (queryText?: string) => {
    const textToSend = (
      queryText !== undefined ? queryText : inputValue
    ).trim();
    if (!textToSend && attachedFiles.length === 0) return;

    const timeStr = new Date().toLocaleTimeString("vi-VN", { hour12: false });
    const userMsgId = `user_${Date.now()}`;

    // Thêm User Message
    const userMessage: ChatMessage = {
      id: userMsgId,
      sender: "user",
      timestamp: timeStr,
      text: textToSend,
      mode,
      attachedFiles: attachedFiles.map((f) => ({
        name: f.name,
        size: formatFileSize(f.size),
      })),
      attachedFileName:
        attachedFiles.length === 1 ? attachedFiles[0].name : undefined,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    const filesToUpload =
      attachedFiles.length > 0 ? [...attachedFiles] : undefined;
    setAttachedFiles([]);
    if (fileInputRef.current) fileInputRef.current.value = "";

    setIsThinking(true);

    try {
      // Gọi API qua ChatService
      const response = await ChatService.solveQuery({
        query: textToSend,
        mode,
        files: filesToUpload,
      });

      const agentTimeStr = new Date().toLocaleTimeString("vi-VN", {
        hour12: false,
      });
      const agentMessage: ChatMessage = {
        id: `agent_${Date.now()}`,
        sender: "agent",
        timestamp: agentTimeStr,
        mode,
        results: response,
      };

      setMessages((prev) => [...prev, agentMessage]);

      // Báo lên component cha để đồng bộ History & Detect CHỈ KHI thành công
      onQueryExecuted?.(textToSend, response, filesToUpload);

      // Cập nhật lại trạng thái cảnh báo nếu có
      performSystemCheck();
    } catch (err: any) {
      console.warn("[Chat Component] Query halted due to error:", err);
      const agentTimeStr = new Date().toLocaleTimeString("vi-VN", {
        hour12: false,
      });

      // Đẩy tin nhắn báo lỗi và dừng tác vụ
      const errorMessage: ChatMessage = {
        id: `agent_err_${Date.now()}`,
        sender: "agent",
        timestamp: agentTimeStr,
        mode,
        isError: true,
        errorMessage:
          err?.message ||
          "Execution halted: Unable to connect to backend engine or no video corpus found.",
      };

      setMessages((prev) => [...prev, errorMessage]);
      performSystemCheck();
    } finally {
      setIsThinking(false);
    }
  };

  // 3. Xử lý xuất file CSV chuẩn theo kết quả thực tế
  const handleExportCSV = (results?: SolveResponse) => {
    if (!results || !results.answers || results.answers.length === 0) return;
    const timestampStr = new Date()
      .toISOString()
      .slice(11, 19)
      .replace(/:/g, "");

    if (mode === "kis") {
      // Format KIS: <video_id>,<frame_id>
      const rows = results.answers.map((ans) => [ans.video_id, ans.frame_id]);
      downloadCSV(`submission_kis_${timestampStr}.csv`, rows);
    } else if (mode === "qna") {
      // Format QNA: <video_id>,<frame_id>,<answer>
      const answerText =
        results.qna_answer?.answer_text || results.answers[0]?.answer || "N/A";
      const rows = results.answers.map((ans) => [
        ans.video_id,
        ans.frame_id,
        ans.answer || answerText,
      ]);
      downloadCSV(`submission_qna_${timestampStr}.csv`, rows);
    } else if (mode === "trake") {
      // Format TRAKE: <video_id>,<frame_1>,...,<frame_N>
      const rows = results.answers.map((ans) => {
        if (ans.milestones && ans.milestones.length > 0) {
          return [ans.video_id, ...ans.milestones.map((m) => m.frameId)];
        }
        return [ans.video_id, ans.frame_id];
      });
      downloadCSV(`submission_trake_${timestampStr}.csv`, rows);
    }
  };

  return (
    <div className="chat-panel" data-component="Chat">
      {/* Ẩn input file cho nút đính kèm */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        style={{ display: "none" }}
        accept="image/*,video/*,.csv,.txt,.json"
      />

      {/* 1. Header Title */}
      <div className="chat-panel__header_01">
        <h1 className="chat-panel__title_01">{getModeTitle()}</h1>
      </div>

      {/* 2. System Warning / Error Banner (Chỉ hiển thị khi có lỗi thực tế) */}
      {systemAlert && (
        <div
          className={`chat-panel__warning-banner_01 chat-panel__warning-banner_01--${systemAlert.level}`}
          data-component="SystemAlertBanner"
        >
          <div className="chat-panel__warning-icon_01">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
              <path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z" />
            </svg>
          </div>
          <div className="chat-panel__warning-content_01">
            <div className="chat-panel__warning-title_01">
              {systemAlert.title}
            </div>
            <div className="chat-panel__warning-desc_01">
              {systemAlert.description}
              {systemAlert.details && (
                <span className="chat-panel__warning-detail_01">
                  {" "}
                  [{systemAlert.details}]
                </span>
              )}
            </div>
          </div>
          <button
            type="button"
            className="chat-panel__warning-retry-btn_01"
            onClick={performSystemCheck}
            title="Kiểm tra lại trạng thái hệ thống"
          >
            RECHECK
          </button>
        </div>
      )}

      {/* 3. Scrollable Conversation Area */}
      <div className="chat-panel__scroll-area_01">
        {messages.length === 0 ? (
          <div className="chat-panel__empty-state_01">
            <div className="chat-panel__empty-icon_01">
              <svg
                width="32"
                height="32"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
              >
                <rect x="2" y="3" width="20" height="14" rx="2" />
                <line x1="8" y1="21" x2="16" y2="21" />
                <line x1="12" y1="17" x2="12" y2="21" />
              </svg>
            </div>
            <span className="chat-panel__empty-title_01">
              SYSTEM_READY // AWAITING_QUERY
            </span>
            <p className="chat-panel__empty-desc_01">
              Enter natural language query or attach reference query file to
              begin retrieval.
            </p>
            <div className="chat-panel__sample-queries_01">
              <span className="chat-panel__sample-label_01">SAMPLE_QUERY:</span>
              <button
                type="button"
                className="chat-panel__sample-btn_01"
                onClick={() => {
                  const q =
                    mode === "qna"
                      ? "What object did the subject place into the trunk?"
                      : mode === "trake"
                        ? "Person enters room THEN opens safe THEN leaves"
                        : "Find video with red car crossing bridge";
                  setInputValue(q);
                }}
              >
                {mode === "qna"
                  ? '"What object did the subject place into the trunk?"'
                  : mode === "trake"
                    ? '"Person enters room THEN opens safe THEN leaves"'
                    : '"Find video with red car crossing bridge"'}
              </button>
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`chat-panel__message-wrapper_01 chat-panel__message-wrapper_01--${
                msg.isError ? "system" : msg.sender
              }`}
            >
              <div
                className={`chat-panel__bubble_01 chat-panel__bubble_01--${
                  msg.isError ? "error" : msg.sender
                }`}
              >
                <div className="chat-panel__bubble-header_01">
                  <div className="chat-panel__sender_01">
                    {msg.isError ? (
                      <svg
                        width="14"
                        height="14"
                        viewBox="0 0 24 24"
                        fill="currentColor"
                      >
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
                      </svg>
                    ) : msg.sender === "user" ? (
                      <svg
                        width="14"
                        height="14"
                        viewBox="0 0 24 24"
                        fill="currentColor"
                      >
                        <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" />
                      </svg>
                    ) : (
                      <svg
                        width="14"
                        height="14"
                        viewBox="0 0 24 24"
                        fill="currentColor"
                      >
                        <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.38-1 1.72V7h4a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h4V5.72c-.6-.34-1-.98-1-1.72a2 2 0 0 1 2-2zM7.5 13a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zm9 0a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zM9 16.5h6v1.5H9v-1.5z" />
                      </svg>
                    )}
                    <span>
                      {msg.isError
                        ? "SYSTEM_ERROR // HALTED"
                        : msg.sender === "user"
                          ? "USER_QUERY"
                          : mode === "qna"
                            ? "QNA_SYNTHESIZER"
                            : mode === "trake"
                              ? "TRAKE_SEQUENCE_ENGINE"
                              : "KIS_ENGINE"}
                    </span>
                  </div>
                  <span className="chat-panel__time_01">{msg.timestamp}</span>
                </div>

                {/* User Content */}
                {msg.sender === "user" && !msg.isError && (
                  <div className="chat-panel__bubble-text_01">
                    {msg.attachedFiles && msg.attachedFiles.length > 0 && (
                      <div className="chat-panel__attached-files-list_01">
                        {msg.attachedFiles.map((af, fIdx) => (
                          <span
                            key={fIdx}
                            className="chat-panel__attached-file-pill_01"
                          >
                            🎬 {af.name} ({af.size})
                          </span>
                        ))}
                      </div>
                    )}
                    {msg.text}
                  </div>
                )}

                {/* Agent Error Content */}
                {msg.isError && (
                  <div className="chat-panel__error-box_01">
                    <p className="chat-panel__error-msg_01">
                      {msg.errorMessage}
                    </p>
                    <div className="chat-panel__error-hint_01">
                      ⚠️ Task stopped. Please ensure Backend Engine (:8000) is
                      running and video corpus is indexed.
                    </div>
                  </div>
                )}

                {/* Agent Response Content */}
                {msg.sender === "agent" && !msg.isError && msg.results && (
                  <>
                    {/* A. Response Cho Mode KIS */}
                    {mode === "kis" && (
                      <>
                        <div className="chat-panel__bubble-text_01">
                          Analysis complete. Ranked{" "}
                          <strong>{msg.results.n_answers}</strong> keyframe
                          candidates based on vector similarity.
                        </div>

                        <div className="chat-panel__results-grid_01">
                          {msg.results.answers.map((frame, idx) => (
                            <div
                              key={idx}
                              className="chat-panel__frame-card_01"
                            >
                              <div className="chat-panel__frame-image_01">
                                {frame.image_url ? (
                                  <img
                                    src={frame.image_url}
                                    alt={`Frame ${frame.frame_id}`}
                                  />
                                ) : (
                                  <div className="chat-panel__frame-placeholder_01" />
                                )}
                                <span className="chat-panel__confidence-badge_01">
                                  {frame.confidence || "90%"}
                                </span>
                              </div>
                              <div className="chat-panel__frame-footer_01">
                                <span className="chat-panel__frame-id_01">
                                  FRM_{frame.frame_id}
                                </span>
                                <button
                                  type="button"
                                  className="chat-panel__download-btn_01"
                                  aria-label={`Download Frame ${frame.frame_id}`}
                                  onClick={() =>
                                    downloadFrameImage(
                                      `FRM_${frame.frame_id}`,
                                      frame.image_url,
                                    )
                                  }
                                >
                                  <svg
                                    width="12"
                                    height="12"
                                    viewBox="0 0 24 24"
                                    fill="none"
                                    stroke="currentColor"
                                    strokeWidth="2.5"
                                  >
                                    <line x1="12" y1="4" x2="12" y2="18" />
                                    <polyline points="18 12 12 18 6 12" />
                                  </svg>
                                </button>
                              </div>
                            </div>
                          ))}
                        </div>

                        <div className="chat-panel__actions_01">
                          <button
                            type="button"
                            className="chat-panel__action-btn_01"
                            onClick={() => handleExportCSV(msg.results)}
                          >
                            <svg
                              width="14"
                              height="14"
                              viewBox="0 0 24 24"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="2"
                            >
                              <rect x="3" y="3" width="7" height="7" />
                              <rect x="14" y="3" width="7" height="7" />
                              <rect x="14" y="14" width="7" height="7" />
                              <rect x="3" y="14" width="7" height="7" />
                            </svg>
                            <span>EXPORT METADATA (.CSV)</span>
                          </button>
                        </div>
                      </>
                    )}

                    {/* B. Response Cho Mode QNA */}
                    {mode === "qna" && (
                      <>
                        <div className="chat-panel__qna-answer-box_01">
                          <div className="chat-panel__qna-header_01">
                            <span className="chat-panel__qna-badge_01">
                              FINAL_ANSWER
                            </span>
                            <span className="chat-panel__qna-conf_01">
                              CONFIDENCE:{" "}
                              {msg.results.qna_answer?.confidence || "95%"}
                            </span>
                          </div>
                          <p className="chat-panel__qna-text_01">
                            {msg.results.qna_answer?.answer_text ||
                              "Answer synthesized from visual evidence."}
                          </p>
                          <div className="chat-panel__qna-meta_01">
                            <span>
                              SOURCE_SEGMENT:{" "}
                              {msg.results.qna_answer?.source_segment ||
                                "L26_V469"}
                            </span>
                            <span>
                              INTERVAL:{" "}
                              {msg.results.qna_answer?.interval ||
                                "14:12:00 - 14:12:30"}
                            </span>
                          </div>
                        </div>

                        {msg.results.qna_answer?.evidence_frames && (
                          <div className="chat-panel__evidence-section_01">
                            <div className="chat-panel__evidence-title_01">
                              VISUAL_EVIDENCE (
                              {msg.results.qna_answer.evidence_frames.length}{" "}
                              FRAMES)
                            </div>
                            <div className="chat-panel__evidence-grid_01">
                              {msg.results.qna_answer.evidence_frames.map(
                                (ev, idx) => (
                                  <div
                                    key={idx}
                                    className="chat-panel__evidence-card_01"
                                  >
                                    <div className="chat-panel__frame-image_01">
                                      {ev.image_url ? (
                                        <img src={ev.image_url} alt={ev.id} />
                                      ) : (
                                        <div className="chat-panel__frame-placeholder_01" />
                                      )}
                                      <span className="chat-panel__confidence-badge_01">
                                        {ev.confidence}
                                      </span>
                                    </div>
                                    <div className="chat-panel__frame-footer_01">
                                      <span className="chat-panel__frame-id_01">
                                        {ev.id}
                                      </span>
                                      <span className="chat-panel__evidence-time_01">
                                        {ev.time}
                                      </span>
                                    </div>
                                  </div>
                                ),
                              )}
                            </div>
                          </div>
                        )}

                        <div className="chat-panel__actions_01">
                          <button
                            type="button"
                            className="chat-panel__action-btn_01"
                            onClick={() => handleExportCSV(msg.results)}
                          >
                            <svg
                              width="14"
                              height="14"
                              viewBox="0 0 24 24"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="2"
                            >
                              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                              <polyline points="14 2 14 8 20 8" />
                              <line x1="16" y1="13" x2="8" y2="13" />
                              <line x1="16" y1="17" x2="8" y2="17" />
                            </svg>
                            <span>EXPORT QNA REPORT (.CSV)</span>
                          </button>
                        </div>
                      </>
                    )}

                    {/* C. Response Cho Mode TRAKE */}
                    {mode === "trake" && (
                      <>
                        <div className="chat-panel__bubble-text_01">
                          Temporal sequence identified across timeline
                          intervals.
                        </div>

                        <div className="chat-panel__timeline-flow_01">
                          {(
                            msg.results.answers[0]?.milestones || [
                              {
                                stepId: "01",
                                stepName: "ENTER_ROOM",
                                frameId: "1012",
                                timestamp: "14:20:05",
                                confidence: "95%",
                              },
                              {
                                stepId: "02",
                                stepName: "OPEN_SAFE",
                                frameId: "1140",
                                timestamp: "14:21:40",
                                confidence: "91%",
                              },
                              {
                                stepId: "03",
                                stepName: "HURRY_EXIT",
                                frameId: "1215",
                                timestamp: "14:22:15",
                                confidence: "89%",
                              },
                            ]
                          ).map((m, idx, arr) => (
                            <div
                              key={idx}
                              className="chat-panel__timeline-item-wrapper_01"
                            >
                              <div className="chat-panel__timeline-card_01">
                                <div className="chat-panel__frame-image_01">
                                  <div className="chat-panel__frame-placeholder_01" />
                                  <span className="chat-panel__confidence-badge_01">
                                    {m.confidence}
                                  </span>
                                </div>
                                <div className="chat-panel__timeline-info_01">
                                  <div className="chat-panel__timeline-name_01">
                                    {m.stepId}. {m.stepName}
                                  </div>
                                  <div className="chat-panel__timeline-meta_01">
                                    <span>FRM_{m.frameId}</span>
                                    <span>{m.timestamp}</span>
                                  </div>
                                </div>
                              </div>

                              {idx < arr.length - 1 && (
                                <div className="chat-panel__timeline-connector_01">
                                  <span>THEN</span>
                                  <svg
                                    width="14"
                                    height="14"
                                    viewBox="0 0 24 24"
                                    fill="none"
                                    stroke="currentColor"
                                    strokeWidth="2.5"
                                  >
                                    <line x1="5" y1="12" x2="19" y2="12" />
                                    <polyline points="12 5 19 12 12 19" />
                                  </svg>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>

                        <div className="chat-panel__actions_01">
                          <button
                            type="button"
                            className="chat-panel__action-btn_01"
                            onClick={() => handleExportCSV(msg.results)}
                          >
                            <svg
                              width="14"
                              height="14"
                              viewBox="0 0 24 24"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="2"
                            >
                              <rect x="3" y="3" width="7" height="7" />
                              <rect x="14" y="3" width="7" height="7" />
                              <rect x="14" y="14" width="7" height="7" />
                              <rect x="3" y="14" width="7" height="7" />
                            </svg>
                            <span>EXPORT TEMPORAL LOG (.CSV)</span>
                          </button>
                        </div>
                      </>
                    )}
                  </>
                )}
              </div>
            </div>
          ))
        )}

        {/* Loading / Thinking Bubble */}
        {isThinking && (
          <div className="chat-panel__message-wrapper_01 chat-panel__message-wrapper_01--system">
            <div className="chat-panel__bubble_01 chat-panel__bubble_01--system">
              <div className="chat-panel__bubble-header_01">
                <div className="chat-panel__sender_01">
                  <span className="chat-panel__thinking-dot_01">●</span>
                  <span>ENGINE_PROCESSING</span>
                </div>
              </div>
              <div className="chat-panel__thinking-text_01">
                Synthesizing dense keyframe vectors & candidate alignment...
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 4. Bottom Command Input */}
      <div className="chat-panel__input-section_01">
        {/* Badge tóm tắt danh sách file đã đính kèm */}
        {attachedFiles.length > 0 && (
          <div className="chat-panel__attached-bar_01">
            <button
              type="button"
              className="chat-panel__attached-summary-btn_01"
              onClick={() => setIsFilesModalOpen(true)}
              title="Click to view and manage uploaded videos"
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="currentColor"
              >
                <path d="M18 4l2 4h-3l-2-4h-2l2 4h-3l-2-4H8l2 4H7L5 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V4h-4z" />
              </svg>
              <span>
                {attachedFiles.length === 1
                  ? `${attachedFiles[0].name} (${formatFileSize(attachedFiles[0].size)})`
                  : `${attachedFiles.length} VIDEOS / FILES ATTACHED`}
              </span>
              <span className="chat-panel__view-tag_01">[VIEW]</span>
            </button>

            <button
              type="button"
              className="chat-panel__remove-all-files-btn_01"
              aria-label="Remove all attached files"
              onClick={handleClearAllFiles}
              title="Xóa tất cả file đính kèm"
            >
              ✕ CLEAR
            </button>
          </div>
        )}

        <div className="chat-panel__input-wrapper_01">
          <span className="chat-panel__input-badge_01">CMD_INPUT</span>
          <div className="chat-panel__input-box_01">
            {/* Nút đính kèm nhiều file */}
            <button
              type="button"
              className="chat-panel__attach-btn_01"
              aria-label="Attach query files or videos"
              onClick={handleTriggerFileInput}
              title="Đính kèm nhiều Video hoặc file truy vấn (Batch Video Upload)"
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="currentColor"
              >
                <path d="M16.5 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5a2.5 2.5 0 0 1 5 0v10.5c0 .83-.67 1.5-1.5 1.5s-1.5-.67-1.5-1.5V6H9v9.5a3 3 0 0 0 6 0V5c0-2.21-1.79-4-4-4S7 2.79 7 5v12.5c0 3.04 2.46 5.5 5.5 5.5s5.5-2.46 5.5-5.5V6h-1.5z" />
              </svg>
            </button>

            <input
              type="text"
              className="chat-panel__text-input_01"
              placeholder={getPlaceholder()}
              value={inputValue}
              disabled={isThinking}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  handleSendQuery();
                }
              }}
            />

            <button
              type="button"
              className="chat-panel__send-btn_01"
              aria-label="Send command"
              disabled={isThinking}
              onClick={() => handleSendQuery()}
            >
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="currentColor"
              >
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
              </svg>
            </button>
          </div>
        </div>

        {/* Utility Toolbar */}
        <div className="chat-panel__toolbar_01">
          <button
            type="button"
            className="chat-panel__tool-btn_01"
            onClick={handleTriggerFileInput}
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            <span>Upload Videos / Batch</span>
          </button>

          {attachedFiles.length > 0 && (
            <button
              type="button"
              className="chat-panel__tool-btn_01"
              onClick={() => setIsFilesModalOpen(true)}
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="currentColor"
              >
                <path d="M18 4l2 4h-3l-2-4h-2l2 4h-3l-2-4H8l2 4H7L5 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V4h-4z" />
              </svg>
              <span>Manage Videos ({attachedFiles.length})</span>
            </button>
          )}

          <button type="button" className="chat-panel__tool-btn_01">
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <line x1="4" y1="21" x2="4" y2="14" />
              <line x1="4" y1="10" x2="4" y2="3" />
              <line x1="12" y1="21" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12" y2="3" />
              <line x1="20" y1="21" x2="20" y2="16" />
              <line x1="20" y1="12" x2="20" y2="3" />
              <line x1="1" y1="14" x2="7" y2="14" />
              <line x1="9" y1="8" x2="15" y2="8" />
              <line x1="17" y1="16" x2="23" y2="16" />
            </svg>
            <span>Filter Settings</span>
          </button>

          <button
            type="button"
            className="chat-panel__tool-btn_01"
            onClick={() => setMessages([])}
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <polyline points="3 6 5 6 21 6" />
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
            </svg>
            <span>Clear Stream</span>
          </button>
        </div>
      </div>

      {/* 5. POPUP MODAL: QUẢN LÝ VÀ XEM TRƯỚC DANH SÁCH VIDEO ĐÃ TẢI LÊN */}
      {isFilesModalOpen && (
        <div
          className="chat-panel__modal-backdrop_01"
          onClick={() => setIsFilesModalOpen(false)}
          data-component="UploadedVideosModal"
        >
          <div
            className="chat-panel__modal_01"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="chat-panel__modal-header_01">
              <div className="chat-panel__modal-title-group_01">
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                >
                  <path d="M18 4l2 4h-3l-2-4h-2l2 4h-3l-2-4H8l2 4H7L5 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V4h-4z" />
                </svg>
                <h3 className="chat-panel__modal-title_01">
                  ATTACHED VIDEOS & QUERY FILES ({attachedFiles.length})
                </h3>
              </div>
              <button
                type="button"
                className="chat-panel__modal-close-btn_01"
                onClick={() => setIsFilesModalOpen(false)}
                aria-label="Close modal"
              >
                ✕
              </button>
            </div>

            {/* Modal Body */}
            <div className="chat-panel__modal-body_01">
              {attachedFiles.length === 0 ? (
                <div className="chat-panel__modal-empty_01">
                  <span className="chat-panel__modal-empty-title_01">
                    NO_VIDEOS_ATTACHED
                  </span>
                  <p className="chat-panel__modal-empty-desc_01">
                    Click "+ ADD MORE VIDEOS" to attach query videos or
                    reference files.
                  </p>
                </div>
              ) : (
                <div className="chat-panel__modal-grid_01">
                  {attachedFiles.map((file, idx) => (
                    <div key={idx} className="chat-panel__modal-card_01">
                      <div className="chat-panel__modal-card-icon_01">
                        {file.type.startsWith("video/") ||
                        file.name.match(/\.(mp4|avi|mov|mkv)$/i) ? (
                          <svg
                            width="22"
                            height="22"
                            viewBox="0 0 24 24"
                            fill="currentColor"
                          >
                            <path d="M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z" />
                          </svg>
                        ) : (
                          <svg
                            width="22"
                            height="22"
                            viewBox="0 0 24 24"
                            fill="currentColor"
                          >
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z" />
                          </svg>
                        )}
                      </div>
                      <div className="chat-panel__modal-card-info_01">
                        <span
                          className="chat-panel__modal-filename_01"
                          title={file.name}
                        >
                          {file.name}
                        </span>
                        <div className="chat-panel__modal-meta_01">
                          <span className="chat-panel__modal-tag_01">
                            {file.name.split(".").pop()?.toUpperCase() ||
                              "FILE"}
                          </span>
                          <span className="chat-panel__modal-size_01">
                            {formatFileSize(file.size)}
                          </span>
                        </div>
                      </div>
                      <button
                        type="button"
                        className="chat-panel__modal-card-remove_01"
                        onClick={() => handleRemoveSingleFile(idx)}
                        title="Xóa video này"
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div className="chat-panel__modal-footer_01">
              <div className="chat-panel__modal-footer-left_01">
                <button
                  type="button"
                  className="chat-panel__modal-action-btn_01"
                  onClick={handleTriggerFileInput}
                >
                  + ADD MORE VIDEOS
                </button>
                {attachedFiles.length > 0 && (
                  <button
                    type="button"
                    className="chat-panel__modal-clear-btn_01"
                    onClick={handleClearAllFiles}
                  >
                    CLEAR ALL
                  </button>
                )}
              </div>
              <button
                type="button"
                className="chat-panel__modal-confirm-btn_01"
                onClick={() => setIsFilesModalOpen(false)}
              >
                DONE ({attachedFiles.length})
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Chat;
