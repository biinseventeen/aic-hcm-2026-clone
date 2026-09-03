/**
 * AIC 2026 Submission Export Utilities
 * Tuân thủ tuyệt đối quy định trong SUBMISSION.md:
 * - Định dạng CSV: UTF-8, dấu phẩy `,`, không header, tối đa 100 dòng.
 * - Textual KIS: <video_id>,<frame_id>
 * - Q&A: <video_id>,<frame_id>,<answer>
 * - TRAKE: <video_id>,<frame_id_1>,...,<frame_id_N>
 */

/**
 * Tải về file CSV theo chuẩn cuộc thi AIC 2026
 */
export function downloadCSV(filename: string, rows: (string | number)[][]): void {
  // Format từng dòng: nếu chuỗi có chứa dấu phẩy hoặc dấu ngoặc kép thì bọc trong ngoặc kép
  const csvContent = rows
    .map((row) =>
      row
        .map((cell) => {
          const str = String(cell);
          if (str.includes(",") || str.includes('"') || str.includes("\n")) {
            return `"${str.replace(/"/g, '""')}"`;
          }
          return str;
        })
        .join(",")
    )
    .join("\r\n");

  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  triggerDownload(filename.endsWith(".csv") ? filename : `${filename}.csv`, blob);
}

/**
 * Tải về file JSON metadata
 */
export function downloadJSON(filename: string, data: object): void {
  const jsonContent = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonContent], { type: "application/json;charset=utf-8;" });
  triggerDownload(filename.endsWith(".json") ? filename : `${filename}.json`, blob);
}

/**
 * Tải về snapshot khung hình (hoặc tạo placeholder image nếu chưa có ảnh thật)
 */
export function downloadFrameImage(frameId: string, imageSrc?: string): void {
  if (imageSrc && !imageSrc.startsWith("blob:")) {
    const link = document.createElement("a");
    link.href = imageSrc;
    link.download = `${frameId}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    return;
  }

  // Tạo ảnh canvas tạm thời có chứa Frame ID và watermark
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 360;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.fillStyle = "#111111";
    ctx.fillRect(0, 0, 640, 360);
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 24px 'Space Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText(`AIC_2026 // ${frameId}`, 320, 180);
    ctx.font = "14px 'Space Mono', monospace";
    ctx.fillStyle = "#888888";
    ctx.fillText("TIMESTAMPED_KEYFRAME_SNAPSHOT", 320, 215);

    canvas.toBlob((blob) => {
      if (blob) {
        triggerDownload(`${frameId}.png`, blob);
      }
    });
  }
}

/**
 * Hàm phụ trợ kích hoạt tải file trên trình duyệt
 */
function triggerDownload(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
