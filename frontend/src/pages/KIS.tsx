import { useState, useEffect } from "react";
import Header from "../components/Header";
import History from "../components/history";
import Chat from "../components/chat";
import Detect from "../components/detect";
import { HistoryService } from "../services/historyService";
import { DetectService } from "../services/detectService";
import { HistoryRecord, DetectionRecord, SolveResponse } from "../services/types";
import "../styles/KIS.css";

function Kis() {
  const [historyItems, setHistoryItems] = useState<HistoryRecord[]>([]);
  const [detectItems, setDetectItems] = useState<DetectionRecord[]>([]);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [isDetectLoading, setIsDetectLoading] = useState(false);

  // Tải lịch sử ban đầu
  useEffect(() => {
    setIsHistoryLoading(true);
    HistoryService.getHistory("kis")
      .then((data) => setHistoryItems(data))
      .catch(() => setHistoryItems([]))
      .finally(() => setIsHistoryLoading(false));
  }, []);

  // Xử lý khi có truy vấn mới từ Chat component
  const handleQueryExecuted = async (query: string, response: SolveResponse) => {
    // 1. Lưu vào bảng History
    const newRecord = await HistoryService.saveHistoryItem({
      query,
      mode: "kis",
      hits: response.n_answers || 0,
      isActive: true,
    });
    setHistoryItems((prev) => [
      newRecord,
      ...prev.map((h) => ({ ...h, isActive: false })),
    ]);

    // 2. Cập nhật bảng Detect bằng dữ liệu thật của kết quả top-1.
    const firstAnswer = response.answers?.[0];
    if (!firstAnswer) {
      setDetectItems([]);
      return;
    }

    setIsDetectLoading(true);
    DetectService.getDetections(firstAnswer.video_id, firstAnswer.frame_id)
      .then((dets) => setDetectItems(dets))
      .catch(() => setDetectItems([]))
      .finally(() => setIsDetectLoading(false));
  };

  return (
    <div className="kis-page" data-component="KisPage">
      <Header />
      <main className="kis-page__workspace_01">
        <History
          mode="kis"
          items={historyItems}
          isLoading={isHistoryLoading}
          onSelectItem={(item) => {
            setHistoryItems((prev) =>
              prev.map((h) => ({ ...h, isActive: h.id === item.id }))
            );
          }}
        />
        <Chat mode="kis" onQueryExecuted={handleQueryExecuted} />
        <Detect items={detectItems} isLoading={isDetectLoading} />
      </main>
    </div>
  );
}

export default Kis;


