import { ApiClient } from "./apiClient";
import { HistoryRecord, RetrievalMode } from "./types";

const LOCAL_STORAGE_KEY = "aic_history_records";

export class HistoryService {
  /**
   * Lấy danh sách lịch sử theo mode (từ API hoặc Local Cache)
   */
  public static async getHistory(mode?: RetrievalMode): Promise<HistoryRecord[]> {
    try {
      // Endpoint Backend trong tương lai: GET /history?mode=kis
      const endpoint = mode ? `/history?mode=${mode}` : "/history";
      return await ApiClient.get<HistoryRecord[]>(endpoint, { timeoutMs: 3000 });
    } catch {
      // Sử dụng local cache
      const stored = localStorage.getItem(LOCAL_STORAGE_KEY);
      if (!stored) return [];
      try {
        const all: HistoryRecord[] = JSON.parse(stored);
        return mode ? all.filter((item) => item.mode === mode) : all;
      } catch {
        return [];
      }
    }
  }

  /**
   * Lưu một query mới vào lịch sử
   */
  public static async saveHistoryItem(item: Omit<HistoryRecord, "id" | "timestamp">): Promise<HistoryRecord> {
    const newRecord: HistoryRecord = {
      ...item,
      id: `hist_${Date.now()}`,
      timestamp: new Date().toLocaleDateString("ja-JP").replace(/\//g, ".") + " AT " + new Date().toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" }),
    };

    try {
      // Endpoint Backend trong tương lai: POST /history
      await ApiClient.post<HistoryRecord>("/history", newRecord, { timeoutMs: 3000 });
    } catch {
      // Lưu vào local cache
      const stored = localStorage.getItem(LOCAL_STORAGE_KEY);
      let all: HistoryRecord[] = [];
      if (stored) {
        try {
          all = JSON.parse(stored);
        } catch {
          all = [];
        }
      }
      all = [newRecord, ...all.filter((h) => h.id !== newRecord.id)].slice(0, 50);
      localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(all));
    }

    return newRecord;
  }

  /**
   * Xóa một mục lịch sử
   */
  public static async deleteHistoryItem(id: string): Promise<boolean> {
    try {
      await ApiClient.get(`/history/delete/${id}`, { timeoutMs: 3000 });
    } catch {
      const stored = localStorage.getItem(LOCAL_STORAGE_KEY);
      if (stored) {
        try {
          const all: HistoryRecord[] = JSON.parse(stored);
          const filtered = all.filter((item) => item.id !== id);
          localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(filtered));
        } catch {
          // ignore
        }
      }
    }
    return true;
  }
}
