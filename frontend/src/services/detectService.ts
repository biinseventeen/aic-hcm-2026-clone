import { ApiClient } from "./apiClient";
import { DetectionRecord } from "./types";

export class DetectService {
  /**
   * Load real organiser-supplied detections for the current result frame.
   * The backend maps an arbitrary source frame to the nearest supplied keyframe.
   */
  public static async getDetections(
    videoId: string,
    frameId: number | string,
  ): Promise<DetectionRecord[]> {
    if (!videoId || frameId === undefined || frameId === null) {
      return [];
    }

    const params = new URLSearchParams({
      video_id: videoId,
      frame_id: String(frameId),
    });

    try {
      return await ApiClient.get<DetectionRecord[]>(
        `/detections?${params.toString()}`,
        { timeoutMs: 5000 },
      );
    } catch (error) {
      console.warn("[DetectService] Unable to load detections:", error);
      return [];
    }
  }
}
