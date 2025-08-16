"use client";
import React from "react";

type Props = {
  age: number;
  children: number;
  occupation: number;
  style?: "casual" | "formal" | "creative";
};

export function DynamicAvatar({ age, children, occupation, style = "casual" }: Props) {
  const base = style === "formal" ? "bg-slate-800" : style === "creative" ? "bg-fuchsia-600" : "bg-sky-600";
  const size = Math.min(120, 72 + children * 6);
  const border = occupation >= 4 ? "ring-4 ring-amber-400" : "ring-2 ring-white";

  return (
    <div className="flex items-center gap-4">
      <div
        className={`grid place-items-center rounded-full ${border} text-white`}
        style={{ width: size, height: size, background: "linear-gradient(135deg,#4f46e5,#06b6d4)" }}
      >
        <div className="text-center text-xs">
          <div className="text-lg font-bold">{age}</div>
          <div>kids:{children}</div>
          <div>occ:{occupation}</div>
        </div>
      </div>
      <div className="rounded-md border px-3 py-2 text-xs text-gray-700">
        Style: {style} <span className={`ml-2 inline-block h-3 w-3 rounded-full ${base}`} />
      </div>
    </div>
  );
}

export default DynamicAvatar;


