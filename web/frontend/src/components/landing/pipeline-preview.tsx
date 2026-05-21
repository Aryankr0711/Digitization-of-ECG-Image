"use client";

import { motion } from "framer-motion";
import { Scan, Grid3X3, Activity, ArrowRight, CheckCircle2 } from "lucide-react";
import { staggerContainer, staggerItem } from "@/animations/variants";

const STAGES = [
  {
    icon: Scan,
    stage: "Stage 0",
    title: "Marker & Orientation Detection",
    description: "Identifies lead labels and determines image orientation using a ResNet18 encoder with UNet decoder.",
    details: ["13 lead markers detected", "8-class orientation", "96% confidence"],
    color: "var(--accent)",
  },
  {
    icon: Grid3X3,
    stage: "Stage 1",
    title: "Grid Structure Detection",
    description: "Detects the ECG grid structure — gridpoints, horizontal and vertical lines — for coordinate calibration.",
    details: ["44 horizontal lines", "57 vertical lines", "Sub-pixel accuracy"],
    color: "var(--cyan)",
  },
  {
    icon: Activity,
    stage: "Stage 2",
    title: "Pixel-Level Segmentation",
    description: "CoordUNet with positional embeddings performs pixel-wise signal extraction across all 4 ECG strips.",
    details: ["Coordinate-aware decoder", "4-channel output", "JS divergence loss"],
    color: "var(--signal-green)",
  },
];

export function PipelinePreview() {
  return (
    <section id="pipeline" className="relative py-28 px-6">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
          className="text-center mb-20"
        >
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight mb-4">
            Three-Stage{" "}
            <span className="text-[var(--accent)]">Pipeline</span>
          </h2>
          <p className="text-[var(--text-secondary)] max-w-xl mx-auto">
            Each stage progressively refines the understanding of the ECG image before signal extraction.
          </p>
        </motion.div>

        <motion.div
          variants={staggerContainer}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-80px" }}
          className="relative"
        >
          {/* Vertical connector line */}
          <div className="absolute left-8 top-0 bottom-0 w-px bg-gradient-to-b from-[var(--accent)]/30 via-[var(--cyan)]/30 to-[var(--signal-green)]/30 hidden md:block" />

          {STAGES.map((stage, index) => (
            <motion.div
              key={stage.stage}
              variants={staggerItem}
              className="relative flex gap-8 mb-12 last:mb-0"
            >
              {/* Node dot */}
              <div className="hidden md:flex flex-col items-center">
                <motion.div
                  whileInView={{ scale: [0.8, 1.1, 1] }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: index * 0.2 }}
                  className="relative z-10 flex items-center justify-center w-16 h-16 rounded-2xl border border-white/10"
                  style={{ backgroundColor: `color-mix(in srgb, ${stage.color} 10%, transparent)` }}
                >
                  <stage.icon className="w-7 h-7" style={{ color: stage.color }} />
                </motion.div>
              </div>

              {/* Content card */}
              <div className="flex-1 card-elevated p-6 md:p-8">
                <div className="flex items-center gap-3 mb-3">
                  <span
                    className="text-xs font-bold font-mono tracking-widest px-2.5 py-0.5 rounded-md"
                    style={{
                      color: stage.color,
                      backgroundColor: `color-mix(in srgb, ${stage.color} 10%, transparent)`,
                    }}
                  >
                    {stage.stage.toUpperCase()}
                  </span>
                  <h3 className="text-lg font-semibold text-[var(--text-primary)]">{stage.title}</h3>
                </div>
                <p className="text-sm text-[var(--text-secondary)] mb-4 leading-relaxed">
                  {stage.description}
                </p>
                <div className="flex flex-wrap gap-3">
                  {stage.details.map((detail) => (
                    <span
                      key={detail}
                      className="flex items-center gap-1.5 text-xs text-[var(--text-tertiary)] font-mono"
                    >
                      <CheckCircle2 className="w-3.5 h-3.5" style={{ color: stage.color }} />
                      {detail}
                    </span>
                  ))}
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
