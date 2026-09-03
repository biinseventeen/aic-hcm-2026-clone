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

      // Backend hiện tại nhận JSON theo contract của aic.api:
      //   text, task, query_id, ...
      // Nó KHÔNG nhận multipart/form-data ở /solve.
      if (allFiles.length > 0) {
        throw new Error(
          "FILE_UPLOAD_UNSUPPORTED: Backend /solve currently accepts JSON queries only."
        );
      }

      return await ApiClient.post<SolveResponse>("/solve", {
        text: req.query,
        task: req.mode,
        query_id: req.query_id || `q_${Date.now()}`,
        ...req.params,
      });
    } catch (err: any) {
      console.warn("[ChatService]: Query failed:", err);

      const message = String(err?.message || "");

      if (
        message.includes("Failed to fetch") ||
        message.includes("NetworkError") ||
        message.includes("Load failed")
      ) {
        throw new Error(
          "BACKEND_UNREACHABLE: Browser cannot access Backend Engine at http://localhost:8000. " +
            "If /health is 200 in the server log, check CORS configuration."
        );
      }

      throw new Error(
        `RETRIEVAL_ERROR: ${message || "Execution stopped due to engine error."}`
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

      // 3. Backend exposes engine warnings under health.engine.warnings.
      const warnings = health.engine?.warnings ?? health.warnings ?? [];
      if (health.status === "degraded" || warnings.length > 0) {
        const warningStr = warnings.join(" ").toLowerCase();
        const corpusProblem =
          warningStr.includes("index") ||
          warningStr.includes("video") ||
          warningStr.includes("corpus") ||
          warningStr.includes("dense");

        if (corpusProblem) {
          return {
            type: "no_video",
            title: "NO_VIDEO_IMPORTED_",
            description:
              "Video corpus or dense vector index is not usable. Check the backend /health details.",
            level: "warning",
            details: warnings.join("; "),
          };
        }
      }

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



}
