"use client";
import React from "react";

type Props = {
  value: number; // 0-100
  label?: string;
};

export function GaugeChart({ value, label = "Score" }: Props) {
  const clamped = Math.max(0, Math.min(100, value));
  const radius = 60;
  const circumference = Math.PI * radius;
  const offset = circumference * (1 - clamped / 100);

  return (
    <div className="flex w-full flex-col items-center">
      <svg width={180} height={110} viewBox="0 0 180 110">
        <g transform="translate(10,10)">
          <path
            d={`M 0 ${radius} A ${radius} ${radius} 0 0 1 ${radius * 2} ${radius}`}
            fill="none"
            stroke="#E5E7EB"
            strokeWidth={14}
          />
          <path
            d={`M 0 ${radius} A ${radius} ${radius} 0 0 1 ${radius * 2} ${radius}`}
            fill="none"
            stroke="#2563EB"
            strokeWidth={14}
            strokeDasharray={`${circumference} ${circumference}`}
            strokeDashoffset={offset}
            strokeLinecap="round"
          />
          <text x={radius} y={radius + 20} textAnchor="middle" className="fill-gray-800" fontSize="14">
            {label}
          </text>
          <text x={radius} y={radius - 5} textAnchor="middle" className="fill-gray-900" fontSize="22" fontWeight={700}>
            {clamped}
          </text>
        </g>
      </svg>
    </div>
  );
}

export default GaugeChart;


