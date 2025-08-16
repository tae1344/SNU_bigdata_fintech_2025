import React from 'react';

interface GaugeProps {
  value: number;
  label: string;
}

const COLORS = {
  primary: "#2563eb",
  text: {
    primary: "#1e40af",
    secondary: "#3730a3",
    light: "#6366f1"
  }
};

export default function LazyGauge({ value, label }: GaugeProps) {
  const clamped = Math.max(0, Math.min(100, value));
  const hue = 120 - (clamped * 1.2); // green -> red
  
  return (
    <div className="text-center" role="region" aria-label={`${label}: ${clamped}%`}>
      <div className="relative w-24 h-24 mx-auto mb-2">
        <svg viewBox="0 0 36 36" className="w-full h-full" aria-hidden="true">
          <path 
            className="opacity-20" 
            strokeWidth="4" 
            stroke="currentColor" 
            fill="none" 
            d="M18 2 a 16 16 0 0 1 0 32 a 16 16 0 0 1 0 -32" 
            style={{ color: COLORS.text.primary }}
          />
          <path 
            strokeWidth="4" 
            stroke={`hsl(${hue}, 80%, 50%)`} 
            fill="none" 
            strokeLinecap="round"
            d={`M18 2 a 16 16 0 0 1 0 32`} 
            style={{ strokeDasharray: `${clamped}, 100` }} 
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="font-bold text-lg" style={{ color: COLORS.text.primary }}>
            {clamped}%
          </span>
        </div>
      </div>
      <div className="text-sm font-medium" style={{ color: COLORS.text.secondary }}>{label}</div>
    </div>
  );
}
