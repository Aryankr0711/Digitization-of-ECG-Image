"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Loader2, ArrowRight, RotateCcw } from "lucide-react";
import { Dropzone } from "@/components/upload/dropzone";
import { ProcessingTimeline } from "@/components/upload/processing-timeline";
import { useUpload } from "@/hooks/use-upload";
import { useWebSocket } from "@/hooks/use-websocket";
import { api } from "@/lib/api";
import type { ProcessingEvent } from "@/types/ecg";
import { fadeInUp } from "@/animations/variants";

export default function UploadPage() {
  const router = useRouter();
  const { file, preview, uploading, uploadProgress, jobId, error, selectFile, upload, reset } = useUpload();

  const [processing, setProcessing] = useState(false);
  const [currentStage, setCurrentStage] = useState<string | null>(null);
  const [stageProgress, setStageProgress] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [wsUrl, setWsUrl] = useState<string | null>(null);

  const handleMessage = useCallback((event: ProcessingEvent) => {
    if (event.stage) setCurrentStage(event.stage);
    if (event.progress !== undefined) setStageProgress(event.progress);
    if (event.message) setLogs((prev) => [...prev, event.message!]);
  }, []);

  const handleComplete = useCallback((resultsUrl: string) => {
    setCurrentStage("complete");
    setStageProgress(1);
    setLogs((prev) => [...prev, "✓ Pipeline complete — redirecting to results..."]);
    // Navigate to results after brief delay
    setTimeout(() => {
      if (jobId) router.push(`/results/${jobId}`);
    }, 1500);
  }, [jobId, router]);

  const handleError = useCallback((error: string) => {
    setLogs((prev) => [...prev, `✗ Error: ${error}`]);
    setProcessing(false);
  }, []);

  useWebSocket({
    url: wsUrl,
    onMessage: handleMessage,
    onComplete: handleComplete,
    onError: handleError,
  });

  const handleStartProcessing = async () => {
    if (!jobId) return;
    setProcessing(true);
    setLogs(["Connecting to pipeline..."]);
    setCurrentStage(null);
    setStageProgress(0);

    // Connect WebSocket first
    setWsUrl(api.getWsUrl(jobId));

    // Small delay to let WS connect, then trigger processing
    await new Promise((r) => setTimeout(r, 500));

    try {
      await api.startProcessing(jobId);
      setLogs((prev) => [...prev, "Pipeline started successfully"]);
    } catch (err) {
      setLogs((prev) => [...prev, `Error: ${err}`]);
      setProcessing(false);
    }
  };

  const handleReset = () => {
    reset();
    setProcessing(false);
    setCurrentStage(null);
    setStageProgress(0);
    setLogs([]);
    setWsUrl(null);
  };

  return (
    <div className="min-h-screen px-6 py-12">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <motion.div
          variants={fadeInUp}
          initial="hidden"
          animate="visible"
          className="mb-10"
        >
          <h1 className="text-3xl font-bold tracking-tight mb-2">Upload ECG Image</h1>
          <p className="text-[var(--text-secondary)]">
            Upload a printed ECG image to extract digital signals through our AI pipeline.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
          {/* Left: Upload area */}
          <div className="lg:col-span-3 space-y-6">
            <Dropzone
              onFileSelect={selectFile}
              preview={preview}
              file={file}
              error={error}
              onClear={handleReset}
            />

            {/* Upload / Process button */}
            {file && !jobId && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                <button
                  onClick={upload}
                  disabled={uploading}
                  className="w-full flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl bg-[var(--accent)] text-[var(--bg-primary)] font-semibold text-sm glow-accent hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                >
                  {uploading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Uploading... {uploadProgress}%
                    </>
                  ) : (
                    <>
                      Upload & Continue
                      <ArrowRight className="w-4 h-4" />
                    </>
                  )}
                </button>
              </motion.div>
            )}

            {jobId && !processing && currentStage !== "complete" && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                <button
                  onClick={handleStartProcessing}
                  className="w-full flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl bg-[var(--accent)] text-[var(--bg-primary)] font-semibold text-sm glow-accent hover:brightness-110 transition-all"
                >
                  Start Processing
                  <ArrowRight className="w-4 h-4" />
                </button>
              </motion.div>
            )}

            {currentStage === "complete" && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-3">
                <button
                  onClick={() => jobId && router.push(`/results/${jobId}`)}
                  className="flex-1 flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl bg-[var(--accent)] text-[var(--bg-primary)] font-semibold text-sm glow-accent"
                >
                  View Results
                  <ArrowRight className="w-4 h-4" />
                </button>
                <button
                  onClick={handleReset}
                  className="flex items-center justify-center gap-2 px-4 py-3.5 rounded-xl border border-[var(--border-medium)] text-[var(--text-secondary)] text-sm hover:text-[var(--text-primary)] transition-colors"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
              </motion.div>
            )}
          </div>

          {/* Right: Processing timeline */}
          <div className="lg:col-span-2">
            <ProcessingTimeline
              currentStage={currentStage}
              stageProgress={stageProgress}
              logs={logs}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
