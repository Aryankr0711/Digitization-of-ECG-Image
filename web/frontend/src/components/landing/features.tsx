"use client";

import { motion } from "framer-motion";
import { Scan, Grid3X3, Activity, Zap, Shield, Download } from "lucide-react";
import { staggerContainer, staggerItem } from "@/animations/variants";

const FEATURES = [
  {
    icon: Scan,
    title: "Intelligent Detection",
    description: "Automatic lead label identification and image orientation detection using deep learning markers.",
  },
  {
    icon: Grid3X3,
    title: "Grid Analysis",
    description: "Precise ECG grid structure detection with sub-pixel accuracy for calibration.",
  },
  {
    icon: Activity,
    title: "12-Lead Extraction",
    description: "Full 12-lead signal extraction including I, II, III, aVR, aVL, aVF, V1-V6, and rhythm.",
  },
  {
    icon: Zap,
    title: "Real-Time Processing",
    description: "GPU-accelerated pipeline with WebSocket streaming for real-time progress updates.",
  },
  {
    icon: Shield,
    title: "Clinical Accuracy",
    description: "Wavelet-based denoising preserves QRS morphology while removing high-frequency noise.",
  },
  {
    icon: Download,
    title: "Export Ready",
    description: "Download extracted signals as CSV, visualization as PNG, or full metrics as JSON.",
  },
];

export function Features() {
  return (
    <section className="relative py-28 px-6">
      <div className="absolute inset-0 gradient-radial-center pointer-events-none" />

      <div className="relative max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight mb-4">
            End-to-End{" "}
            <span className="text-[var(--accent)]">Intelligence</span>
          </h2>
          <p className="text-[var(--text-secondary)] max-w-xl mx-auto">
            A multi-stage deep learning pipeline that transforms paper ECGs into clinically accurate digital signals.
          </p>
        </motion.div>

        <motion.div
          variants={staggerContainer}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-80px" }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5"
        >
          {FEATURES.map((feature) => (
            <motion.div
              key={feature.title}
              variants={staggerItem}
              whileHover={{ y: -4, transition: { duration: 0.2 } }}
              className="group card-elevated p-6"
            >
              <div className="flex items-center gap-4 mb-4">
                <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-[var(--accent)]/[0.08] border border-[var(--accent)]/10 group-hover:border-[var(--accent)]/25 transition-colors">
                  <feature.icon className="w-5 h-5 text-[var(--accent)]" />
                </div>
                <h3 className="text-base font-semibold text-[var(--text-primary)]">{feature.title}</h3>
              </div>
              <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
