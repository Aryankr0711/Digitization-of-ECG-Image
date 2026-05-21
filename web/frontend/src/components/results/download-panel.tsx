"use client";

import { motion } from "framer-motion";
import { Download, FileSpreadsheet, FileImage, FileJson } from "lucide-react";
import { api } from "@/lib/api";

interface DownloadPanelProps {
  jobId: string;
}

const DOWNLOAD_OPTIONS = [
  { type: "csv" as const, label: "CSV Signals", description: "All 13 leads as CSV", icon: FileSpreadsheet, color: "var(--accent)" },
  { type: "json" as const, label: "JSON Metrics", description: "Quality metrics data", icon: FileJson, color: "var(--cyan)" },
  { type: "image" as const, label: "Original Image", description: "Source ECG image", icon: FileImage, color: "var(--signal-blue)" },
];

export function DownloadPanel({ jobId }: DownloadPanelProps) {
  const handleDownload = (type: "csv" | "json" | "image" | "png") => {
    const url = api.getDownloadUrl(jobId, type);
    window.open(url, "_blank");
  };

  return (
    <div className="card-elevated p-5">
      <div className="flex items-center gap-2 mb-4">
        <Download className="w-4 h-4 text-[var(--text-tertiary)]" />
        <h3 className="text-sm font-semibold text-[var(--text-primary)]">Downloads</h3>
      </div>
      <div className="space-y-2">
        {DOWNLOAD_OPTIONS.map((option) => (
          <motion.button
            key={option.type}
            whileHover={{ x: 2 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => handleDownload(option.type)}
            className="w-full flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-white/[0.03] transition-colors text-left"
          >
            <div
              className="flex items-center justify-center w-9 h-9 rounded-lg"
              style={{ backgroundColor: `color-mix(in srgb, ${option.color} 10%, transparent)` }}
            >
              <option.icon className="w-4 h-4" style={{ color: option.color }} />
            </div>
            <div>
              <p className="text-sm font-medium text-[var(--text-primary)]">{option.label}</p>
              <p className="text-xs text-[var(--text-tertiary)]">{option.description}</p>
            </div>
          </motion.button>
        ))}
      </div>
    </div>
  );
}
