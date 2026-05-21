"use client";
import { useState, useCallback } from "react";
import { api } from "@/lib/api";
import type { UploadResponse } from "@/types/ecg";

interface UseUploadReturn {
  file: File | null;
  preview: string | null;
  uploading: boolean;
  uploadProgress: number;
  jobId: string | null;
  error: string | null;
  selectFile: (file: File) => void;
  upload: () => Promise<void>;
  reset: () => void;
}

const ALLOWED_TYPES = ["image/png", "image/jpeg", "image/jpg"];
const MAX_SIZE = 50 * 1024 * 1024; // 50MB

export function useUpload(): UseUploadReturn {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const selectFile = useCallback((selectedFile: File) => {
    setError(null);

    if (!ALLOWED_TYPES.includes(selectedFile.type)) {
      setError("Invalid file type. Please upload a PNG, JPG, or JPEG file.");
      return;
    }
    if (selectedFile.size > MAX_SIZE) {
      setError("File too large. Maximum size is 50MB.");
      return;
    }

    setFile(selectedFile);
    // Generate preview
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(selectedFile);
  }, []);

  const upload = useCallback(async () => {
    if (!file) return;
    setUploading(true);
    setUploadProgress(0);
    setError(null);

    try {
      // Simulate upload progress (fetch doesn't expose upload progress)
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => Math.min(prev + 10, 90));
      }, 200);

      const response: UploadResponse = await api.uploadImage(file);
      clearInterval(progressInterval);
      setUploadProgress(100);
      setJobId(response.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }, [file]);

  const reset = useCallback(() => {
    setFile(null);
    setPreview(null);
    setUploading(false);
    setUploadProgress(0);
    setJobId(null);
    setError(null);
  }, []);

  return {
    file,
    preview,
    uploading,
    uploadProgress,
    jobId,
    error,
    selectFile,
    upload,
    reset,
  };
}
