"use client";

import { motion } from "framer-motion";
import { TrendingUp, Activity, BarChart3, Timer } from "lucide-react";
import { formatMetric, getQualityColor, getQualityLabel, formatDuration } from "@/lib/utils";
import type { Metrics } from "@/types/ecg";
import { staggerContainer, staggerItem } from "@/animations/variants";

interface MetricsPanelProps {
  metrics: Metrics;
  processingTime?: number;
}

export function MetricsPanel({ metrics, processingTime }: MetricsPanelProps) {
  const cards = [
    {
      label: "Signal-to-Noise",
      value: `${metrics.snr_db.toFixed(1)}`,
      unit: "dB",
      icon: TrendingUp,
      quality: getQualityLabel(metrics.snr_db),
      qualityColor: getQualityColor(metrics.snr_db),
      accent: "var(--accent)",
    },
    {
      label: "RMSE",
      value: formatMetric(metrics.rmse),
      unit: "mV",
      icon: Activity,
      accent: "var(--cyan)",
    },
    {
      label: "MAE",
      value: formatMetric(metrics.mae),
      unit: "mV",
      icon: BarChart3,
      accent: "var(--signal-blue)",
    },
    {
      label: "Processing Time",
      value: processingTime ? formatDuration(processingTime) : "—",
      unit: "",
      icon: Timer,
      accent: "var(--signal-amber)",
    },
  ];

  return (
    <motion.div
      variants={staggerContainer}
      initial="hidden"
      animate="visible"
      className="grid grid-cols-2 lg:grid-cols-4 gap-4"
    >
      {cards.map((card) => (
        <motion.div
          key={card.label}
          variants={staggerItem}
          className="card-elevated p-5"
        >
          <div className="flex items-center gap-2 mb-3">
            <card.icon className="w-4 h-4" style={{ color: card.accent }} />
            <span className="text-xs text-[var(--text-tertiary)] font-medium">{card.label}</span>
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="text-2xl font-bold font-mono text-[var(--text-primary)] metric-value">
              {card.value}
            </span>
            {card.unit && (
              <span className="text-xs text-[var(--text-tertiary)] font-mono">{card.unit}</span>
            )}
          </div>
          {card.quality && (
            <p className={`text-xs font-medium mt-2 ${card.qualityColor}`}>
              {card.quality}
            </p>
          )}
        </motion.div>
      ))}
    </motion.div>
  );
}
