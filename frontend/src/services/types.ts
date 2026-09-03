/**
 * Data Transfer Objects (DTO) & API Types for AIC 2026 Frontend-Backend Contract
 * Tuân thủ theo đặc tả BACKEND.md & SUBMISSION.md
 */

export type RetrievalMode = "kis" | "qna" | "trake";

// 1. DTO cho Backend API: POST /solve
export interface SolveRequest {
  query: string;
  mode: RetrievalMode;
  query_id?: string;
  file?: File;
  files?: File[];
  top_k?: number;
  params?: Record<string, any>;
}


export interface SolveAnswerItem {
  rank: number;
  video_id: string;
  frame_id: number | string;
  gain?: number;
  cumulative?: number;
  source?: string;
  confidence?: string | number;
  answer?: string; // Dành cho Q&A mode
  milestones?: {
    // Dành cho TRAKE mode
    stepId: string;
    stepName: string;
    frameId: number | string;
    timestamp: string;
    confidence: string;
  }[];
  image_url?: string;
  timestamp?: string;
  timestamp_seconds?: number;
  preview_frame_id?: number;
  preview_keyframe_n?: number;
}

export interface SolveResponse {
  degraded?: boolean;
  n_answers: number;
  allocation?: {
    expected_final?: number;
  };
  answers: SolveAnswerItem[];
  qna_answer?: {
    answer_text: string;
    confidence: string;
    source_segment: string;
    interval?: string;
    evidence_frames?: {
      id: string;
      frame_id: number | string;
      time: string;
      confidence: string;
      image_url?: string;
    }[];
  };
  message?: string;
  raw_log?: any;
}

// 2. DTO cho Backend API: GET /health
export interface HealthResponse {
  status: "ready" | "degraded" | "loading" | "error";
  degraded?: boolean;
  warnings?: string[];
  version?: string;
  detail?: string;
  engine?: {
    ready?: boolean;
    warnings?: string[];
    n_videos?: number;
    n_keyframes?: number;
    [key: string]: any;
  };
}

// 3. DTO cho Bảng History (Lịch sử truy vấn)
export interface HistoryRecord {
  id: string;
  timestamp: string;
  query: string;
  mode: RetrievalMode;
  hits: number;
  isActive?: boolean;
}

// 4. DTO cho Bảng Detect (Nhận diện đối tượng)
export interface DetectTagItem {
  label: string;
  variant?: "solid" | "text";
}

export interface DetectionRecord {
  id: string;
  videoSource: string;
  frameId: string;
  requestedFrameId?: string;
  timestamp: string;
  confidence: string;
  isAlert?: boolean;
  imageSrc?: string;
  tags: DetectTagItem[];
  isSelected?: boolean;
  bbox?: [number, number, number, number]; // [ymin, xmin, ymax, xmax]
}

// 5. Chat Message State
export interface ChatMessage {
  id: string;
  sender: "user" | "agent";
  timestamp: string;
  text?: string;
  mode: RetrievalMode;
  results?: SolveResponse;
  isLoading?: boolean;
  isError?: boolean;
  errorMessage?: string;
  attachedFileName?: string;
  attachedFiles?: { name: string; size: string }[];
}



// 6. System Alert / Error Cases
export type AlertType = "network_offline" | "backend_disconnected" | "no_video";

export interface SystemAlert {
  type: AlertType;
  title: string;
  description: string;
  level: "warning" | "error";
  details?: string;
}

