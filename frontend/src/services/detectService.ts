import { ApiClient } from "./apiClient";
import { DetectionRecord } from "./types";

export class DetectService {
  /**
   * Lấy danh sách đối tượng nhận diện từ backend theo queryId hoặc videoId
   */
  public static async getDetections(videoId?: string, queryId?: string): Promise<DetectionRecord[]> {
    try {
      // Endpoint Backend trong tương lai: GET /detections?video_id=...&query_id=...
      const params = new URLSearchParams();
      if (videoId) params.append("video_id", videoId);
      if (queryId) params.append("query_id", queryId);

      const endpoint = `/detections?${params.toString()}`;
      return await ApiClient.get<DetectionRecord[]>(endpoint, { timeoutMs: 3000 });
    } catch {
      // Trả về dữ liệu scan mẫu cho frame hiện tại
      return [
        {
          id: "det_1",
          videoSource: videoId || "VID_C_02",
          frameId: "4492-A",
          timestamp: "14:12:33",
          confidence: "94%",
          isAlert: true,
          tags: [
            { label: "PERSON", variant: "solid" },
            { label: "RED_BAG", variant: "solid" },
          ],
          isSelected: true,
        },
        {
          id: "det_2",
          videoSource: videoId || "VID_C_02",
          frameId: "4510-B",
          timestamp: "14:13:05",
          confidence: "88%",
          tags: [
            { label: "PERSON", variant: "text" },
            { label: "RED_BAG", variant: "solid" },
          ],
        },
        {
          id: "det_3",
          videoSource: "VID_C_05",
          frameId: "1022-X",
          timestamp: "14:22:11",
          confidence: "82%",
          tags: [{ label: "PARTIAL_MATCH", variant: "text" }],
        },
      ];
    }
  }
}
