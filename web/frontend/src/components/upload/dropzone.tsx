"use client";

import { useCallback } from "react";
import { motion } from "framer-motion";
import { Upload, Image, X, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface DropzoneProps {
  onFileSelect: (file: File) => void;
  preview: string | null;
  file: File | null;
  error: string | null;
  onClear: () => void;
}

export function Dropzone({ onFileSelect, preview, file, error, onClear }: DropzoneProps) {
  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile) onFileSelect(droppedFile);
    },
    [onFileSelect]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selected = e.target.files?.[0];
      if (selected) onFileSelect(selected);
    },
    [onFileSelect]
  );

  if (preview && file) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        className="relative card-elevated overflow-hidden"
      >
        {/* Image preview */}
        <div className="relative aspect-[16/7] bg-[var(--bg-primary)]">
          <img
            src={preview}
            alt="ECG preview"
            className="w-full h-full object-contain p-4"
          />
          {/* Gradient overlay at bottom */}
          <div className="absolute bottom-0 left-0 right-0 h-20 bg-gradient-to-t from-[var(--bg-card)] to-transparent" />
        </div>

        {/* File info bar */}
        <div className="flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-[var(--accent)]/10 flex items-center justify-center">
              <Image className="w-5 h-5 text-[var(--accent)]" />
            </div>
            <div>
              <p className="text-sm font-medium text-[var(--text-primary)] truncate max-w-xs">{file.name}</p>
              <p className="text-xs text-[var(--text-tertiary)] font-mono">
                {(file.size / (1024 * 1024)).toFixed(2)} MB · {file.type.split("/")[1].toUpperCase()}
              </p>
            </div>
          </div>
          <button
            onClick={onClear}
            className="p-2 rounded-lg text-[var(--text-tertiary)] hover:text-[var(--text-primary)] hover:bg-white/5 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <label
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        className={cn(
          "group relative flex flex-col items-center justify-center w-full min-h-[320px] rounded-2xl border-2 border-dashed cursor-pointer transition-all duration-300",
          error
            ? "border-red-500/40 bg-red-500/[0.03]"
            : "border-[var(--border-medium)] hover:border-[var(--accent)]/40 bg-[var(--bg-card)]/50 hover:bg-[var(--accent)]/[0.02]"
        )}
      >
        <input
          type="file"
          accept=".png,.jpg,.jpeg"
          onChange={handleFileInput}
          className="hidden"
        />

        <motion.div
          whileHover={{ scale: 1.05 }}
          className="flex items-center justify-center w-16 h-16 rounded-2xl bg-[var(--accent)]/[0.08] border border-[var(--accent)]/15 mb-5 group-hover:glow-accent transition-all duration-300"
        >
          <Upload className="w-7 h-7 text-[var(--accent)]" />
        </motion.div>

        <p className="text-base font-medium text-[var(--text-primary)] mb-2">
          Drop your ECG image here
        </p>
        <p className="text-sm text-[var(--text-secondary)] mb-4">
          or click to browse files
        </p>
        <p className="text-xs text-[var(--text-tertiary)] font-mono">
          PNG, JPG, JPEG · Max 50MB
        </p>

        {error && (
          <motion.div
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center gap-2 mt-4 px-4 py-2 rounded-lg bg-red-500/10 text-red-400 text-sm"
          >
            <AlertCircle className="w-4 h-4" />
            {error}
          </motion.div>
        )}
      </label>
    </motion.div>
  );
}
