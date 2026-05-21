"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { LEAD_NAMES } from "@/types/ecg";

interface LeadSelectorProps {
  activeLead: string | "all";
  onSelect: (lead: string | "all") => void;
}

const LIMB_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF"];
const PRECORDIAL_LEADS = ["V1", "V2", "V3", "V4", "V5", "V6"];

export function LeadSelector({ activeLead, onSelect }: LeadSelectorProps) {
  return (
    <div className="card-elevated p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-semibold text-[var(--text-tertiary)] tracking-widest">LEADS</h3>
      </div>

      <div className="space-y-3">
        {/* All leads button */}
        <button
          onClick={() => onSelect("all")}
          className={cn(
            "w-full px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 text-left",
            activeLead === "all"
              ? "bg-[var(--accent)]/10 text-[var(--accent)] border border-[var(--accent)]/20"
              : "text-[var(--text-secondary)] hover:bg-white/[0.03] border border-transparent"
          )}
        >
          All Leads (13)
        </button>

        {/* Separator */}
        <div className="h-px bg-[var(--border-subtle)]" />

        {/* Limb leads */}
        <div>
          <p className="text-[10px] font-semibold text-[var(--text-tertiary)] tracking-widest mb-1.5 px-1">LIMB</p>
          <div className="grid grid-cols-3 gap-1">
            {LIMB_LEADS.map((lead) => (
              <button
                key={lead}
                onClick={() => onSelect(lead)}
                className={cn(
                  "relative px-2 py-1.5 rounded-lg text-xs font-mono font-medium transition-all duration-200",
                  activeLead === lead
                    ? "bg-[var(--accent)]/10 text-[var(--accent)]"
                    : "text-[var(--text-secondary)] hover:bg-white/[0.03]"
                )}
              >
                {lead}
                {activeLead === lead && (
                  <motion.div
                    layoutId="lead-indicator"
                    className="absolute inset-0 rounded-lg border border-[var(--accent)]/25"
                    transition={{ type: "spring", bounce: 0.2, duration: 0.4 }}
                  />
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Precordial leads */}
        <div>
          <p className="text-[10px] font-semibold text-[var(--text-tertiary)] tracking-widest mb-1.5 px-1">PRECORDIAL</p>
          <div className="grid grid-cols-3 gap-1">
            {PRECORDIAL_LEADS.map((lead) => (
              <button
                key={lead}
                onClick={() => onSelect(lead)}
                className={cn(
                  "relative px-2 py-1.5 rounded-lg text-xs font-mono font-medium transition-all duration-200",
                  activeLead === lead
                    ? "bg-[var(--accent)]/10 text-[var(--accent)]"
                    : "text-[var(--text-secondary)] hover:bg-white/[0.03]"
                )}
              >
                {lead}
                {activeLead === lead && (
                  <motion.div
                    layoutId="lead-indicator"
                    className="absolute inset-0 rounded-lg border border-[var(--accent)]/25"
                    transition={{ type: "spring", bounce: 0.2, duration: 0.4 }}
                  />
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Rhythm strip */}
        <div>
          <p className="text-[10px] font-semibold text-[var(--text-tertiary)] tracking-widest mb-1.5 px-1">RHYTHM</p>
          <button
            onClick={() => onSelect("II-rhythm")}
            className={cn(
              "relative w-full px-2 py-1.5 rounded-lg text-xs font-mono font-medium transition-all duration-200 text-left",
              activeLead === "II-rhythm"
                ? "bg-[var(--accent)]/10 text-[var(--accent)]"
                : "text-[var(--text-secondary)] hover:bg-white/[0.03]"
            )}
          >
            II-rhythm
            {activeLead === "II-rhythm" && (
              <motion.div
                layoutId="lead-indicator"
                className="absolute inset-0 rounded-lg border border-[var(--accent)]/25"
                transition={{ type: "spring", bounce: 0.2, duration: 0.4 }}
              />
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
