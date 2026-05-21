"use client";

import { motion } from "framer-motion";
import { Check, Loader2, Circle, Scan, Grid3X3, Activity, GitBranch, BarChart3, CheckCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ProcessingEvent } from "@/types/ecg";

const STAGE_CONFIG = [
  { id: "stage_0_marker_detection", name: "Marker Detection", icon: Scan, description: "Detecting lead labels & orientation" },
  { id: "stage_1_grid_detection", name: "Grid Detection", icon: Grid3X3, description: "Analyzing ECG grid structure" },
  { id: "stage_2_signal_segmentation", name: "Signal Segmentation", icon: Activity, description: "Pixel-level signal extraction" },
  { id: "extracting_leads", name: "Lead Extraction", icon: GitBranch, description: "Extracting 12-lead signals" },
  { id: "generating_metrics", name: "Quality Metrics", icon: BarChart3, description: "Computing accuracy metrics" },
  { id: "complete", name: "Complete", icon: CheckCircle, description: "Pipeline finished" },
];

interface ProcessingTimelineProps {
  currentStage: string | null;
  stageProgress: number;
  logs: string[];
}

export function ProcessingTimeline({ currentStage, stageProgress, logs }: ProcessingTimelineProps) {
  const currentIndex = STAGE_CONFIG.findIndex((s) => s.id === currentStage);

  return (
    <div className="space-y-6">
      {/* Stage timeline */}
      <div className="card-elevated p-6">
        <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-5 tracking-wide">PROCESSING PIPELINE</h3>
        <div className="space-y-1">
          {STAGE_CONFIG.map((stage, index) => {
            const isComplete = index < currentIndex || currentStage === "complete";
            const isActive = stage.id === currentStage && currentStage !== "complete";
            const isPending = index > currentIndex;

            return (
              <div key={stage.id} className="relative">
                <div className={cn(
                  "flex items-center gap-4 px-4 py-3 rounded-xl transition-all duration-300",
                  isActive && "bg-[var(--accent)]/[0.06]",
                )}>
                  {/* Status icon */}
                  <div className="relative flex-shrink-0">
                    {isComplete ? (
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        className="w-8 h-8 rounded-full bg-[var(--accent)]/15 flex items-center justify-center"
                      >
                        <Check className="w-4 h-4 text-[var(--accent)]" />
                      </motion.div>
                    ) : isActive ? (
                      <div className="w-8 h-8 rounded-full bg-[var(--accent)]/15 flex items-center justify-center">
                        <Loader2 className="w-4 h-4 text-[var(--accent)] animate-spin" />
                      </div>
                    ) : (
                      <div className="w-8 h-8 rounded-full bg-white/[0.03] flex items-center justify-center">
                        <Circle className="w-3 h-3 text-[var(--text-tertiary)]" />
                      </div>
                    )}
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className={cn(
                        "text-sm font-medium",
                        isComplete ? "text-[var(--accent)]" :
                        isActive ? "text-[var(--text-primary)]" :
                        "text-[var(--text-tertiary)]"
                      )}>
                        {stage.name}
                      </span>
                    </div>
                    <p className="text-xs text-[var(--text-tertiary)] truncate">{stage.description}</p>
                  </div>

                  {/* Progress */}
                  {isActive && (
                    <span className="text-xs font-mono text-[var(--accent)]">
                      {Math.round(stageProgress * 100)}%
                    </span>
                  )}
                </div>

                {/* Progress bar for active stage */}
                {isActive && (
                  <div className="mx-4 mt-1 mb-2">
                    <div className="h-1 rounded-full bg-white/[0.04] overflow-hidden">
                      <motion.div
                        className="h-full rounded-full bg-gradient-to-r from-[var(--accent)] to-[var(--cyan)]"
                        initial={{ width: 0 }}
                        animate={{ width: `${stageProgress * 100}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Processing console */}
      <div className="card-elevated overflow-hidden">
        <div className="flex items-center gap-2 px-5 py-3 border-b border-[var(--border-subtle)]">
          <div className="flex gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full bg-red-500/60" />
            <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/60" />
            <div className="w-2.5 h-2.5 rounded-full bg-green-500/60" />
          </div>
          <span className="text-xs text-[var(--text-tertiary)] font-mono ml-2">processing.log</span>
        </div>
        <div className="p-4 max-h-48 overflow-y-auto font-mono text-xs leading-6 text-[var(--text-secondary)] bg-[var(--bg-primary)]/60">
          {logs.length === 0 ? (
            <p className="text-[var(--text-tertiary)] italic">Waiting for pipeline to start...</p>
          ) : (
            logs.map((log, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -5 }}
                animate={{ opacity: 1, x: 0 }}
                className="flex gap-3"
              >
                <span className="text-[var(--accent)]/60 select-none">{String(i + 1).padStart(3, "0")}</span>
                <span>{log}</span>
              </motion.div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
