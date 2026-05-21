"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, Sparkles } from "lucide-react";
import { fadeInUp, staggerContainer, staggerItem } from "@/animations/variants";

export function Hero() {
  return (
    <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden">
      {/* Radial gradient overlay */}
      <div className="absolute inset-0 gradient-radial-top pointer-events-none" />

      <motion.div
        variants={staggerContainer}
        initial="hidden"
        animate="visible"
        className="relative z-10 max-w-4xl mx-auto px-6 text-center"
      >
        {/* Badge */}
        <motion.div variants={staggerItem} className="flex justify-center mb-8">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-[var(--accent)]/20 bg-[var(--accent)]/[0.05] text-[var(--accent)] text-xs font-medium tracking-wide">
            <Sparkles className="w-3.5 h-3.5" />
            AI-POWERED ECG DIGITIZATION
          </div>
        </motion.div>

        {/* Headline */}
        <motion.h1
          variants={staggerItem}
          className="text-5xl sm:text-6xl md:text-7xl font-extrabold tracking-tight leading-[1.08] mb-6"
        >
          <span className="block text-[var(--text-primary)]">Transform Printed</span>
          <span className="block text-[var(--text-primary)]">ECGs into</span>
          <span className="block bg-gradient-to-r from-[var(--accent)] via-[var(--cyan)] to-[var(--accent)] bg-clip-text text-transparent text-glow">
            Digital Intelligence
          </span>
        </motion.h1>

        {/* Subtext */}
        <motion.p
          variants={staggerItem}
          className="text-lg sm:text-xl text-[var(--text-secondary)] max-w-2xl mx-auto mb-10 leading-relaxed"
        >
          AI-powered digitization pipeline converting paper ECG images into precise
          multi-lead digital signals with clinical-grade accuracy.
        </motion.p>

        {/* CTA Buttons */}
        <motion.div variants={staggerItem} className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link href="/upload">
            <motion.button
              whileHover={{ scale: 1.03, boxShadow: "0 0 40px rgba(0, 212, 170, 0.25)" }}
              whileTap={{ scale: 0.98 }}
              className="group flex items-center gap-2.5 px-8 py-3.5 rounded-xl bg-[var(--accent)] text-[var(--bg-primary)] font-semibold text-sm transition-all duration-200 glow-accent"
            >
              Upload ECG Image
              <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
            </motion.button>
          </Link>
          <Link href="#pipeline">
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="flex items-center gap-2 px-8 py-3.5 rounded-xl border border-[var(--border-medium)] text-[var(--text-secondary)] font-medium text-sm hover:text-[var(--text-primary)] hover:border-[var(--accent)]/30 transition-all duration-200"
            >
              View Pipeline
            </motion.button>
          </Link>
        </motion.div>

        {/* Stats row */}
        <motion.div
          variants={staggerItem}
          className="flex items-center justify-center gap-8 sm:gap-12 mt-16 pt-8 border-t border-[var(--border-subtle)]"
        >
          {[
            { value: "12", label: "Lead Extraction" },
            { value: "3", label: "Stage Pipeline" },
            { value: ">95%", label: "Accuracy" },
          ].map((stat) => (
            <div key={stat.label} className="text-center">
              <div className="text-2xl sm:text-3xl font-bold text-[var(--accent)] font-mono">{stat.value}</div>
              <div className="text-xs text-[var(--text-tertiary)] mt-1">{stat.label}</div>
            </div>
          ))}
        </motion.div>

        {/* Logos row */}
        <motion.div
          variants={staggerItem}
          className="flex flex-col items-center justify-center mt-12 pt-12 border-t border-[var(--border-subtle)]"
        >
          <p className="text-xs text-[var(--text-tertiary)] uppercase tracking-widest font-semibold mb-6">Powered By</p>
          <div className="flex flex-wrap items-center justify-center gap-12 sm:gap-20 transition-all duration-300">
            <img src="/logos/image.png" alt="Partner Logo 1" className="h-8 md:h-12 object-contain" />
            <img src="/logos/image2.png" alt="Partner Logo 2" className="h-10 md:h-14 object-contain" />
          </div>
        </motion.div>
      </motion.div>
    </section>
  );
}
