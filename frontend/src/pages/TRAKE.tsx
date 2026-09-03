import { useState, useEffect } from "react";
import Header from "../components/Header";
import History from "../components/history";
import Chat from "../components/chat";
import Detect from "../components/detect";
import { HistoryService } from "../services/historyService";
import { DetectService } from "../services/detectService";
import { HistoryRecord, DetectionRecord, SolveResponse } from "../services/types";
import "../styles/KIS.css";

function Trake() {
  const [historyItems, setHistoryItems] = useState<HistoryRecord[]>([]);
  const [detectItems, setDetectItems] = useState<DetectionRecord[]>([]);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [isDetectLoading, setIsDetectLoading] = useState(false);

  useEffect(() => {
    setIsHistoryLoading(true);
    HistoryService.getHistory("trake")
      .then((data) => setHistoryItems(data))
      .catch(() => setHistoryItems([]))
      .finally(() => setIsHistoryLoading(false));
  }, []);

  const handleQueryExecuted = async (query: string, response: SolveResponse) => {
    const newRecord = await HistoryService.saveHistoryItem({
      query,
      mode: "trake",
      hits: response.n_answers || 0,
      isActive: true,
    });
    setHistoryItems((prev) => [
      newRecord,
      ...prev.map((h) => ({ ...h, isActive: false })),
    ]);

    const firstVideo = response.answers?.[0]?.video_id || "VID_C_02";
    setIsDetectLoading(true);
    DetectService.getDetections(firstVideo)
      .then((dets) => setDetectItems(dets))
      .finally(() => setIsDetectLoading(false));
  };

  return (
    <div className="kis-page" data-component="TrakePage">
      <Header />
      <main className="kis-page__workspace_01">
        <History
          mode="trake"
          items={historyItems}
          isLoading={isHistoryLoading}
          onSelectItem={(item) => {
            setHistoryItems((prev) =>
              prev.map((h) => ({ ...h, isActive: h.id === item.id }))
            );
          }}
        />
        <Chat mode="trake" onQueryExecuted={handleQueryExecuted} />
        <Detect items={detectItems} isLoading={isDetectLoading} />
      </main>
    </div>
  );
}

export default Trake;


