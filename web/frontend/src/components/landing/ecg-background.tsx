"use client";

import { useEffect, useRef } from "react";

/**
 * Full-screen animated ECG waveform background.
 * Renders multiple traces at low opacity using requestAnimationFrame.
 */
export function ECGBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationId: number;
    let time = 0;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    const drawECGWave = (
      yBase: number,
      speed: number,
      amplitude: number,
      opacity: number,
      color: string
    ) => {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.globalAlpha = opacity;
      ctx.lineWidth = 1.2;

      for (let x = 0; x < canvas.width; x += 2) {
        const t = (x / canvas.width) * 12 + time * speed;
        const cycle = t % 1.2;
        let y = yBase;

        if (cycle > 0.15 && cycle < 0.25) {
          // P wave
          y -= amplitude * 0.12 * Math.sin(((cycle - 0.15) / 0.1) * Math.PI);
        } else if (cycle > 0.3 && cycle < 0.33) {
          // Q dip
          y += amplitude * 0.08 * Math.sin(((cycle - 0.3) / 0.03) * Math.PI);
        } else if (cycle > 0.33 && cycle < 0.40) {
          // R peak
          y -= amplitude * Math.sin(((cycle - 0.33) / 0.07) * Math.PI);
        } else if (cycle > 0.40 && cycle < 0.44) {
          // S dip
          y += amplitude * 0.12 * Math.sin(((cycle - 0.40) / 0.04) * Math.PI);
        } else if (cycle > 0.55 && cycle < 0.70) {
          // T wave
          y -= amplitude * 0.25 * Math.sin(((cycle - 0.55) / 0.15) * Math.PI);
        }

        if (x === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    };

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      time += 0.003;

      drawECGWave(canvas.height * 0.25, 0.4, 50, 0.06, "#00d4aa");
      drawECGWave(canvas.height * 0.45, 0.3, 40, 0.04, "#06b6d4");
      drawECGWave(canvas.height * 0.65, 0.5, 35, 0.05, "#00d4aa");
      drawECGWave(canvas.height * 0.8, 0.35, 30, 0.03, "#06b6d4");

      animationId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none"
      style={{ willChange: "transform" }}
      aria-hidden="true"
    />
  );
}
