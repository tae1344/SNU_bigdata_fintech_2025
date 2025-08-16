"use client";
import React from "react";

type Scenario = {
  title: string;
  score: number; // 0-100
  probability: number; // 0-1
  details?: Record<string, number | string>;
};

type Props = {
  left: Scenario;
  right: Scenario;
};

export function ComparePanel({ left, right }: Props) {
  return (
    <div className="grid w-full grid-cols-1 gap-4 md:grid-cols-2">
      {[left, right].map((s, idx) => (
        <div key={idx} className="rounded-lg border bg-white p-4 shadow">
          <div className="mb-2 text-lg font-semibold">{s.title}</div>
          <div className="mb-1 text-sm text-gray-600">Score: {s.score}</div>
          <div className="mb-3 text-sm text-gray-600">
            Probability: {(s.probability * 100).toFixed(1)}%
          </div>
          {s.details && (
            <div className="space-y-1 text-sm">
              {Object.entries(s.details).map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <span className="text-gray-700">{k}</span>
                  <span className="text-gray-900">{String(v)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export default ComparePanel;


