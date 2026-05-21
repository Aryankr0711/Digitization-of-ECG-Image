/**
 * API client — typed wrapper around fetch for backend communication.
 */
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

import type { UploadResponse, ECGResults, JobStatus } from "@/types/ecg";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...options?.headers,
    },
  });
  if (!res.ok) {
    const error = await res.text().catch(() => "Request failed");
    throw new Error(`API Error ${res.status}: ${error}`);
  }
  return res.json();
}

export const api = {
  /** Upload an ECG image file */
  async uploadImage(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append("file", file);
    return request<UploadResponse>("/api/upload", {
      method: "POST",
      body: formData,
    });
  },

  /** Start processing a previously uploaded image */
  async startProcessing(jobId: string): Promise<{ message: string; job_id: string }> {
    return request(`/api/process/${jobId}`, { method: "POST" });
  },

  /** Get current job status */
  async getStatus(jobId: string): Promise<JobStatus> {
    return request<JobStatus>(`/api/status/${jobId}`);
  },

  /** Get full results for a completed job */
  async getResults(jobId: string): Promise<ECGResults> {
    return request<ECGResults>(`/api/results/${jobId}`);
  },

  /** Get download URL for a specific file type */
  getDownloadUrl(jobId: string, type: "csv" | "json" | "image" | "png"): string {
    return `${API_BASE}/api/download/${jobId}/${type}`;
  },

  /** Get WebSocket URL for a job */
  getWsUrl(jobId: string): string {
    return `${WS_BASE}/ws/progress/${jobId}`;
  },
};
