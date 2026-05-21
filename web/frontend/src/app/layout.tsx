import type { Metadata } from "next";
import { Navbar } from "@/components/layout/navbar";
import "./globals.css";

export const metadata: Metadata = {
  title: "ECG Digitizer — AI-Powered ECG Image to Signal Conversion",
  description: "Transform printed ECG images into precise multi-lead digital signals using advanced deep learning. Supports 12-lead extraction, real-time processing, and clinical-grade accuracy.",
  keywords: ["ECG", "digitization", "medical AI", "signal processing", "deep learning", "12-lead ECG"],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[var(--bg-primary)] text-[var(--text-primary)] antialiased">
        <Navbar />
        <main className="pt-16">
          {children}
        </main>
      </body>
    </html>
  );
}
