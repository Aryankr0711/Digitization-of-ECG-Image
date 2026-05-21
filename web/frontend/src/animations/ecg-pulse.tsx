"use client";

/**
 * Animated ECG pulse SVG — used in the navbar logo and hero.
 */
export function ECGPulse({ className = "", width = 120, height = 32 }: {
  className?: string;
  width?: number;
  height?: number;
}) {
  // ECG waveform path: flat → P-wave → QRS → T-wave → flat
  const path = `M 0 ${height/2} L ${width*0.15} ${height/2} L ${width*0.2} ${height*0.4} L ${width*0.25} ${height/2} L ${width*0.35} ${height/2} L ${width*0.38} ${height*0.7} L ${width*0.42} ${height*0.05} L ${width*0.46} ${height*0.8} L ${width*0.5} ${height/2} L ${width*0.6} ${height/2} L ${width*0.65} ${height*0.35} L ${width*0.72} ${height/2} L ${width*0.85} ${height/2} L ${width} ${height/2}`;

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className={className}
      aria-hidden="true"
    >
      {/* Glow layer */}
      <path
        d={path}
        fill="none"
        stroke="var(--accent)"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity="0.3"
        filter="blur(4px)"
        className="ecg-animate"
      />
      {/* Main line */}
      <path
        d={path}
        fill="none"
        stroke="var(--accent)"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="ecg-animate"
      />
    </svg>
  );
}
