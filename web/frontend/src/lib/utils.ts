import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Format milliseconds to human-readable duration */
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const seconds = ms / 1000;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${minutes}m ${secs}s`;
}

/** Format a metric value for display */
export function formatMetric(value: number, precision: number = 4): string {
  return value.toFixed(precision);
}

/** Get quality color based on SNR value */
export function getQualityColor(snr: number): string {
  if (snr >= 25) return "text-emerald-400";
  if (snr >= 20) return "text-amber-400";
  return "text-red-400";
}

/** Get quality label based on SNR value */
export function getQualityLabel(snr: number): string {
  if (snr >= 25) return "Excellent";
  if (snr >= 20) return "Good";
  if (snr >= 15) return "Fair";
  return "Poor";
}
