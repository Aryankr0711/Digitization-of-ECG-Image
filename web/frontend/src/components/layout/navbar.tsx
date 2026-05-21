"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { Activity, Upload, BarChart3 } from "lucide-react";
import { ECGPulse } from "@/animations/ecg-pulse";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { href: "/", label: "Home", icon: Activity },
  { href: "/upload", label: "Upload", icon: Upload },
  { href: "/results", label: "Results", icon: BarChart3 },
];

export function Navbar() {
  const pathname = usePathname();

  return (
    <motion.header
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="fixed top-0 left-0 right-0 z-50 glass-strong"
    >
      <nav className="mx-auto max-w-7xl flex items-center justify-between px-6 h-16">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-3 group">
          <div className="relative flex items-center justify-center w-8 h-8 rounded-lg bg-[var(--accent)]/10 border border-[var(--accent)]/20 group-hover:border-[var(--accent)]/40 transition-colors">
            <Activity className="w-4 h-4 text-[var(--accent)]" />
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-semibold tracking-tight text-[var(--text-primary)]">
              ECG Digitizer
            </span>
            <ECGPulse width={72} height={12} className="opacity-60 group-hover:opacity-100 transition-opacity" />
          </div>
        </Link>

        {/* Navigation */}
        <div className="flex items-center gap-1">
          {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
            const isActive = pathname === href || (href !== "/" && pathname.startsWith(href));
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "relative flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200",
                  isActive
                    ? "text-[var(--accent)]"
                    : "text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-white/[0.03]"
                )}
              >
                <Icon className="w-4 h-4" />
                <span>{label}</span>
                {isActive && (
                  <motion.div
                    layoutId="nav-active"
                    className="absolute inset-0 rounded-lg bg-[var(--accent)]/[0.08] border border-[var(--accent)]/20"
                    transition={{ type: "spring", bounce: 0.2, duration: 0.5 }}
                  />
                )}
              </Link>
            );
          })}
        </div>

        {/* Status and Logos */}
        <div className="flex items-center gap-6 hidden md:flex">
          <div className="flex items-center gap-4">
            <img src="/logos/image.png" alt="Partner Logo 1" className="h-6 object-contain" />
            <img src="/logos/image2.png" alt="Partner Logo 2" className="h-8 object-contain" />
          </div>
          <div className="flex items-center gap-2">
            <div className="relative w-2 h-2 rounded-full bg-emerald-400 pulse-dot" />
            <span className="text-xs text-[var(--text-tertiary)] font-mono">ONLINE</span>
          </div>
        </div>
      </nav>

      {/* Bottom glow line */}
      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-[var(--accent)]/20 to-transparent" />
    </motion.header>
  );
}
