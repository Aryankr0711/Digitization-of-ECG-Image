import { ECGBackground } from "@/components/landing/ecg-background";
import { Hero } from "@/components/landing/hero";
import { Features } from "@/components/landing/features";
import { PipelinePreview } from "@/components/landing/pipeline-preview";

export default function HomePage() {
  return (
    <div className="relative">
      <ECGBackground />
      <Hero />
      <Features />
      <PipelinePreview />

      {/* Footer */}
      <footer className="relative border-t border-[var(--border-subtle)] py-12 px-6">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-sm text-[var(--text-tertiary)]">
            ECG Digitizer — AI-Powered Medical Signal Processing
          </p>
          <p className="text-xs text-[var(--text-tertiary)] font-mono">
            PhysioNet Challenge · Deep Learning · UNet Architecture
          </p>
        </div>
      </footer>
    </div>
  );
}
