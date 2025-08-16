"use client";
import React from "react";

export type Contribution = { key: string; value: number };

type Props = {
  contributions: Contribution[];
};

export function ContributionsBar({ contributions }: Props) {
  const top = contributions.slice(0, 3);
  return (
    <div className="w-full space-y-3">
      {top.map((c) => {
        const abs = Math.min(1, Math.abs(c.value));
        const width = `${Math.round(abs * 100)}%`;
        const positive = c.value >= 0;
        return (
          <div key={c.key} className="w-full">
            <div className="mb-1 flex justify-between text-sm text-gray-700">
              <span>{c.key}</span>
              <span className={positive ? "text-green-600" : "text-red-600"}>
                {c.value.toFixed(3)}
              </span>
            </div>
            <div className="h-3 w-full rounded bg-gray-100">
              <div
                className={`h-3 rounded ${positive ? "bg-green-500" : "bg-red-500"}`}
                style={{ width }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default ContributionsBar;


