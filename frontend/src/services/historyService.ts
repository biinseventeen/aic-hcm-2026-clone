import { HistoryRecord, RetrievalMode } from "./types";

const LOCAL_STORAGE_KEY = "aic_history_records";

export class HistoryService {
  /**
   * Backend hiện tại chưa có /history.
   * History là state của frontend nên dùng localStorage trực tiếp,
   * tránh gọi endpoint không tồn tại rồi tạo 404 giả.
   */
  public static async getHistory(mode?: RetrievalMode): Promise<HistoryRecord[]> {
    const stored = localStorage.getItem(LOCAL_STORAGE_KEY);
    if (!stored) return [];

    try {
      const all: HistoryRecord[] = JSON.parse(stored);
      return mode ? all.filter((item) => item.mode === mode) : all;
    } catch {
      return [];
    }
  }

  /**
   * Lưu một query mới vào local history.
   */
  public static async saveHistoryItem(
    item: Omit<HistoryRecord, "id" | "timestamp">
  ): Promise<HistoryRecord> {
    const now = new Date();

    const newRecord: HistoryRecord = {
      ...item,
      id: `hist_${Date.now()}`,
      timestamp:
        now.toLocaleDateString("ja-JP").replace(/\//g, ".") +
        " AT " +
        now.toLocaleTimeString("vi-VN", {
          hour: "2-digit",
          minute: "2-digit",
        }),
    };

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

    return newRecord;
  }

  /**
   * Xóa một mục lịch sử khỏi local history.
   */
  public static async deleteHistoryItem(id: string): Promise<boolean> {
    const stored = localStorage.getItem(LOCAL_STORAGE_KEY);

    if (stored) {
      try {
        const all: HistoryRecord[] = JSON.parse(stored);
        const filtered = all.filter((item) => item.id !== id);
        localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(filtered));
      } catch {
        // Cache hỏng: bỏ qua.
      }
    }

    return true;
  }
}
