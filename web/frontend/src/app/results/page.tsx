"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { BarChart3, ArrowRight, Upload } from "lucide-react";
import { fadeInUp } from "@/animations/variants";

export default function ResultsIndexPage() {
  return (
    <div className="min-h-screen flex items-center justify-center px-6">
      <motion.div
        variants={fadeInUp}
        initial="hidden"
        animate="visible"
        className="flex flex-col items-center text-center max-w-md"
      >
        <div className="w-16 h-16 rounded-2xl bg-[var(--accent)]/[0.08] border border-[var(--accent)]/15 flex items-center justify-center mb-6">
          <BarChart3 className="w-8 h-8 text-[var(--accent)]" />
        </div>
        <h1 className="text-2xl font-bold mb-3">No Results Yet</h1>
        <p className="text-[var(--text-secondary)] text-sm mb-8 leading-relaxed">
          Upload and process an ECG image to view extracted waveforms,
          metrics, and downloadable signals here.
        </p>
        <Link href="/upload">
          <motion.button
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.98 }}
            className="flex items-center gap-2 px-6 py-3 rounded-xl bg-[var(--accent)] text-[var(--bg-primary)] font-semibold text-sm glow-accent"
          >
            <Upload className="w-4 h-4" />
            Upload ECG Image
            <ArrowRight className="w-4 h-4" />
          </motion.button>
        </Link>
      </motion.div>
    </div>
  );
}
