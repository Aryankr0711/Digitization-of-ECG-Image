"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { motion } from "framer-motion";
import { Loader2, AlertCircle, ArrowLeft } from "lucide-react";
import Link from "next/link";
import { api } from "@/lib/api";
import { ECGViewer, ECGGridView } from "@/components/results/ecg-viewer";
import { LeadSelector } from "@/components/results/lead-selector";
import { DownloadPanel } from "@/components/results/download-panel";
import type { ECGResults } from "@/types/ecg";
import { fadeInUp } from "@/animations/variants";

export default function ResultsPage() {
  const params = useParams();
  const jobId = params.id as string;
  const [results, setResults] = useState<ECGResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeLead, setActiveLead] = useState<string | "all">("all");

  useEffect(() => {
    async function fetchResults() {
      try {
        const data = await api.getResults(jobId);
        setResults(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load results");
      } finally {
        setLoading(false);
      }
    }
    if (jobId) fetchResults();
  }, [jobId]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="w-8 h-8 text-[var(--accent)] animate-spin" />
          <p className="text-sm text-[var(--text-secondary)]">Loading results...</p>
        </div>
      </div>
    );
  }

  if (error || !results) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="flex flex-col items-center gap-4 card-elevated p-8 max-w-md text-center">
          <AlertCircle className="w-10 h-10 text-[var(--signal-red)]" />
          <h2 className="text-lg font-semibold">Results Not Available</h2>
          <p className="text-sm text-[var(--text-secondary)]">{error || "Processing may still be in progress."}</p>
          <Link href="/upload" className="flex items-center gap-2 text-sm text-[var(--accent)] hover:underline mt-2">
            <ArrowLeft className="w-4 h-4" /> Back to Upload
          </Link>
        </div>
      </div>
    );
  }

  const selectedLeads = activeLead === "all"
    ? results.leads
    : results.leads.filter((l) => l.name === activeLead);

  return (
    <div className="min-h-screen px-6 py-12">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div variants={fadeInUp} initial="hidden" animate="visible" className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <Link href="/upload" className="text-[var(--text-tertiary)] hover:text-[var(--text-secondary)] transition-colors">
              <ArrowLeft className="w-4 h-4" />
            </Link>
            <h1 className="text-2xl font-bold tracking-tight">ECG Analysis Results</h1>
          </div>
          <p className="text-[var(--text-secondary)] text-sm">
            Job <span className="font-mono text-[var(--accent)]">{jobId}</span> · {results.leads.length} leads extracted
          </p>
        </motion.div>

        {/* Metrics removed per request */}

        {/* Main content area */}
        <div className="grid grid-cols-1 lg:grid-cols-6 gap-6">
          {/* Lead selector sidebar */}
          <div className="lg:col-span-1">
            <div className="lg:sticky lg:top-24 space-y-4">
              <LeadSelector activeLead={activeLead} onSelect={setActiveLead} />
              <DownloadPanel jobId={jobId} />
            </div>
          </div>

          {/* ECG viewer area */}
          <div className="lg:col-span-5">
            {/* Original Image Comparison */}
            <div className="card-elevated p-6 mb-6">
              <h3 className="text-lg font-semibold mb-4 text-[var(--text-primary)]">Original Uploaded ECG</h3>
              <div className="relative w-full overflow-hidden rounded-lg border border-[var(--border-subtle)] bg-white/5">
                <img
                  src={`http://localhost:8000/outputs/${jobId}/original.png`}
                  alt="Original ECG"
                  className="w-full h-auto object-contain max-h-[400px]"
                  onError={(e) => {
                    e.currentTarget.style.display = 'none';
                  }}
                />
              </div>
            </div>

            {activeLead === "all" ? (
              <ECGGridView leads={results.leads} />
            ) : (
              <div className="space-y-4">
                {selectedLeads.map((lead) => (
                  <ECGViewer key={lead.name} lead={lead} height={400} />
                ))}


              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
