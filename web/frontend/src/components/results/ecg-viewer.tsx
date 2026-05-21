"use client";

import { useMemo } from "react";
import { motion } from "framer-motion";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";
import type { ECGLead } from "@/types/ecg";

interface ECGViewerProps {
  lead: ECGLead;
  color?: string;
  showGrid?: boolean;
  height?: number;
  animate?: boolean;
}

export function ECGViewer({
  lead,
  color = "#00d4aa",
  showGrid = true,
  height = 280,
  animate = true,
}: ECGViewerProps) {
  // Transform lead data into chart data points
  const chartData = useMemo(() => {
    const step = lead.sample_rate ? 1 / lead.sample_rate : 0.002; // 500Hz default
    return lead.data.map((value, i) => ({
      time: parseFloat((i * step).toFixed(4)),
      value: parseFloat(value.toFixed(4)),
    }));
  }, [lead.data, lead.sample_rate]);

  // We don't want to heavily downsample because ECG peaks are very narrow.
  // We can downsample by 2 or 3 to keep it performant while preserving shape.
  const displayData = useMemo(() => {
    const maxPoints = 5000;
    if (chartData.length <= maxPoints) return chartData;
    const step = Math.ceil(chartData.length / maxPoints);
    return chartData.filter((_, i) => i % step === 0);
  }, [chartData]);

  const yMin = useMemo(() => Math.min(...lead.data) * 1.2, [lead.data]);
  const yMax = useMemo(() => Math.max(...lead.data) * 1.2, [lead.data]);

  return (
    <motion.div
      initial={animate ? { opacity: 0, y: 10 } : false}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="relative w-full rounded-xl overflow-hidden ecg-grid"
      style={{ backgroundColor: "var(--bg-primary)" }}
    >
      {/* Lead name badge */}
      <div className="absolute top-3 left-4 z-10">
        <span className="text-xs font-bold font-mono px-2.5 py-1 rounded-md bg-[var(--bg-card)] border border-[var(--border-subtle)] text-[var(--text-primary)]">
          {lead.name}
        </span>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={displayData} margin={{ top: 32, right: 24, bottom: 16, left: 8 }}>
          {showGrid && (
            <CartesianGrid
              strokeDasharray="none"
              stroke="rgba(0, 212, 170, 0.04)"
              strokeWidth={0.5}
            />
          )}
          <XAxis
            dataKey="time"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(v) => `${v.toFixed(1)}s`}
            stroke="var(--text-tertiary)"
            fontSize={10}
            fontFamily="var(--font-mono)"
            tickLine={false}
            axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
            minTickGap={30}
            tickMargin={8}
          />
          <YAxis
            domain={[yMin, yMax]}
            tickFormatter={(v) => `${v.toFixed(1)}`}
            stroke="var(--text-tertiary)"
            fontSize={10}
            fontFamily="var(--font-mono)"
            tickLine={false}
            axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
            width={32}
            tickMargin={4}
          />
          <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" strokeWidth={1} strokeDasharray="3 3" />
          <Tooltip
            contentStyle={{
              backgroundColor: "var(--bg-elevated)",
              border: "1px solid var(--border-medium)",
              borderRadius: "8px",
              fontSize: "12px",
              fontFamily: "var(--font-mono)",
              padding: "8px 12px",
            }}
            labelFormatter={(v) => `${Number(v).toFixed(3)}s`}
            formatter={(v) => [`${Number(v).toFixed(4)} mV`, lead.name]}
          />
          <Line
            type="linear"
            dataKey="value"
            stroke={color}
            strokeWidth={1.5}
            dot={false}
            activeDot={{ r: 3, fill: color, stroke: "var(--bg-primary)", strokeWidth: 2 }}
            animationDuration={animate ? 2000 : 0}
            animationEasing="ease-in-out"
            isAnimationActive={animate}
          />
        </LineChart>
      </ResponsiveContainer>
    </motion.div>
  );
}

/**
 * Multi-lead grid view showing all leads simultaneously.
 */
export function ECGGridView({ leads, animate = true }: { leads: ECGLead[]; animate?: boolean }) {
  const colors = [
    "#00d4aa", "#06b6d4", "#22d3ee", "#3b82f6",
    "#8b5cf6", "#d946ef", "#f43f5e", "#f59e0b",
    "#10b981", "#14b8a6", "#00d4aa", "#06b6d4", "#22d3ee",
  ];

  return (
    <div className="flex flex-col gap-8">
      {leads.map((lead, i) => (
        <ECGViewer
          key={lead.name}
          lead={lead}
          color={colors[i % colors.length]}
          height={280}
          animate={animate}
        />
      ))}
    </div>
  );
}
