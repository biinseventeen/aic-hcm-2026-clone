import { ApiClient } from "./apiClient";
import { SolveRequest, SolveResponse, HealthResponse } from "./types";

export class ChatService {
  /**
   * Gọi API POST /solve để giải truy vấn KIS, QNA hoặc TRAKE
   */
  public static async solveQuery(req: SolveRequest): Promise<SolveResponse> {
    // 1. Kiểm tra mạng
    if (typeof navigator !== "undefined" && !navigator.onLine) {
      throw new Error("NETWORK_OFFLINE: No internet/network connection detected.");
    }

    try {
      const allFiles = req.files || (req.file ? [req.file] : []);
      if (allFiles.length > 0) {
        const formData = new FormData();
        formData.append("query", req.query);
        formData.append("mode", req.mode);
        if (req.query_id) formData.append("query_id", req.query_id);
        if (req.top_k) formData.append("top_k", String(req.top_k));

        allFiles.forEach((f) => {
          formData.append("files", f);
        });

        return await ApiClient.post<SolveResponse>("/solve", formData);
      }

      return await ApiClient.post<SolveResponse>("/solve", {
        query: req.query,
        mode: req.mode,
        query_id: req.query_id || `q_${Date.now()}`,
        top_k: req.top_k || 100,
        ...req.params,
      });
    } catch (err: any) {

      // Báo lỗi thực tế thay vì sinh kết quả giả mạo
      console.warn("[ChatService]: Query failed:", err);
      throw new Error(
        err?.message?.includes("Failed to fetch") || err?.message?.includes("NetworkError")
          ? "BACKEND_UNREACHABLE: Cannot connect to Backend Engine at http://localhost:8000. Task halted."
          : `RETRIEVAL_ERROR: ${err?.message || "Execution stopped due to engine error."}`
      );
    }
  }


  /**
   * Kiểm tra tình trạng engine backend: GET /health
   */
  public static async checkHealth(): Promise<HealthResponse> {
    try {
      return await ApiClient.get<HealthResponse>("/health", { timeoutMs: 3000 });
    } catch {
      return {
        status: "error",
        degraded: true,
        warnings: ["Backend server (port 8000) not connected."],
      };
    }
  }

  /**
   * Kiểm tra hệ thống toàn diện để phát hiện lỗi:
   * 1. Mất kết nối mạng (Network Offline)
   * 2. Mất kết nối Backend (Backend Disconnected)
   * 3. Chưa import Video (No Video / Index Missing)
   * Trả về null nếu hệ thống hoàn toàn bình thường (không có lỗi).
   */
  public static async checkSystemStatus(): Promise<import("./types").SystemAlert | null> {
    // 1. Kiểm tra kết nối mạng của trình duyệt
    if (typeof navigator !== "undefined" && !navigator.onLine) {
      return {
        type: "network_offline",
        title: "NETWORK_OFFLINE_",
        description: "No internet/network connection detected. Browser is currently offline.",
        level: "error",
      };
    }

    // 2. Kiểm tra kết nối tới Backend Engine (:8000)
    try {
      const health = await ApiClient.get<HealthResponse>("/health", { timeoutMs: 2500 });

      // 3. Kiểm tra xem đã import video / dense index chưa
      if (health.warnings && health.warnings.length > 0) {
        const warningStr = health.warnings.join(" ").toLowerCase();
        if (
          warningStr.includes("index") ||
          warningStr.includes("video") ||
          warningStr.includes("corpus") ||
          warningStr.includes("dense") ||
          health.status === "degraded"
        ) {
          return {
            type: "no_video",
            title: "NO_VIDEO_IMPORTED_",
            description:
              "Video corpus or dense vector index not found in data/batch1. Please import video data before executing retrieval.",
            level: "warning",
            details: health.warnings.join("; "),
          };
        }
      }

      // Nếu backend sẵn sàng và không có cảnh báo nghiêm trọng
      return null;
    } catch (err: any) {
      // Backend không phản hồi
      return {
        type: "backend_disconnected",
        title: "BACKEND_DISCONNECTED_",
        description:
          "Unable to connect to Engine Façade at http://localhost:8000. Ensure 'uv run aic serve' is running.",
        level: "error",
      };
    }
  }


  /**
   * Fallback engine simulator khi chưa kết nối backend
   */
  private static generateFallbackResponse(req: SolveRequest): SolveResponse {
    const timestampStr = new Date().toLocaleTimeString("vi-VN", { hour12: false });

    if (req.mode === "qna") {
      return {
        degraded: true,
        n_answers: 2,
        allocation: { expected_final: 0.94 },
        answers: [
          { rank: 1, video_id: "L26_V469", frame_id: 4492, confidence: "96%", answer: "black metallic suitcase" },
          { rank: 2, video_id: "L26_V469", frame_id: 4510, confidence: "94%", answer: "black metallic suitcase" },
        ],
        qna_answer: {
          answer_text: `Based on visual analysis of query "${req.query}", the subject placed a black metallic suitcase into the rear cargo compartment.`,
          confidence: "96%",
          source_segment: "L26_V469",
          interval: "14:12:00 - 14:12:30",
          evidence_frames: [
            { id: "FRM_4492-A", frame_id: 4492, time: timestampStr, confidence: "96%" },
            { id: "FRM_4510-B", frame_id: 4510, time: timestampStr, confidence: "94%" },
          ],
        },
      };
    }

    if (req.mode === "trake") {
      return {
        degraded: true,
        n_answers: 3,
        allocation: { expected_final: 0.88 },
        answers: [
          {
            rank: 1,
            video_id: "L26_V469",
            frame_id: 1012,
            confidence: "95%",
            milestones: [
              { stepId: "01", stepName: "ENTER_ROOM", frameId: 1012, timestamp: "14:20:05", confidence: "95%" },
              { stepId: "02", stepName: "OPEN_SAFE", frameId: 1140, timestamp: "14:21:40", confidence: "91%" },
              { stepId: "03", stepName: "HURRY_EXIT", frameId: 1215, timestamp: "14:22:15", confidence: "89%" },
            ],
          },
        ],
      };
    }

    // Default KIS mode
    return {
      degraded: true,
      n_answers: 4,
      allocation: { expected_final: 0.91 },
      answers: [
        { rank: 1, video_id: "L26_V469", frame_id: 372, confidence: "92%" },
        { rank: 2, video_id: "L26_V469", frame_id: 480, confidence: "88%" },
        { rank: 3, video_id: "L26_V469", frame_id: 496, confidence: "85%" },
        { rank: 4, video_id: "L26_V469", frame_id: 542, confidence: "94%" },
      ],
    };
  }
}
