/**
 * TypeScript interfaces for ECG data, API responses, and component props.
 */

export interface ECGLead {
  name: string;
  data: number[];
  sample_rate: number;
}

export interface Metrics {
  snr_db: number;
  rmse: number;
  mae: number;
  mse: number;
}

export interface LeadMetrics {
  lead_name: string;
  metrics: Metrics;
}

export interface ECGResults {
  job_id: string;
  leads: ECGLead[];
  lead_metrics: LeadMetrics[];
  average_metrics: Metrics;
  processing_time_ms: number;
  original_image_url: string;
  stages_completed: string[];
}

export interface UploadResponse {
  job_id: string;
  filename: string;
  message: string;
}

export interface ProcessingEvent {
  type: "progress" | "complete" | "error" | "pong";
  job_id: string;
  stage?: string;
  stage_index?: number;
  total_stages?: number;
  progress?: number;
  message?: string;
  results_url?: string;
  error?: string;
  timestamp?: number;
}

export interface JobStatus {
  job_id: string;
  status: string;
  progress: number;
  current_stage?: string;
  error?: string;
}

export type PipelineStage = {
  id: string;
  name: string;
  description: string;
  icon: string;
  status: "pending" | "active" | "complete" | "error";
  progress: number;
};

export const LEAD_NAMES = [
  "I", "II", "III", "aVR", "aVL", "aVF",
  "V1", "V2", "V3", "V4", "V5", "V6", "II-rhythm",
] as const;

export type LeadName = typeof LEAD_NAMES[number];

export const PIPELINE_STAGES: Omit<PipelineStage, "status" | "progress">[] = [
  { id: "stage_0_marker_detection", name: "Marker Detection", description: "Detecting lead labels & orientation", icon: "Scan" },
  { id: "stage_1_grid_detection", name: "Grid Detection", description: "Analyzing ECG grid structure", icon: "Grid3X3" },
  { id: "stage_2_signal_segmentation", name: "Signal Segmentation", description: "Pixel-level signal extraction", icon: "Activity" },
  { id: "extracting_leads", name: "Lead Extraction", description: "Extracting 12-lead ECG signals", icon: "GitBranch" },
  { id: "generating_metrics", name: "Quality Metrics", description: "Computing SNR, RMSE, MAE", icon: "BarChart3" },
  { id: "complete", name: "Complete", description: "Pipeline finished", icon: "CheckCircle" },
];
