"use client";
import React from "react";

type Props = {
  probability: number; // 0-1
};

export function AvatarScene({ probability }: Props) {
  const p = Math.max(0, Math.min(1, probability));
  const distance = 80 + 240 * p; // px
  const speed = 2000 - 1000 * p; // ms
  const effectLevel = 1 + Math.floor(2 * p);

  return (
    <div className="relative h-40 w-full overflow-hidden rounded border bg-white">
      <div
        className="absolute bottom-4 left-4 h-10 w-10 rounded-full bg-indigo-500 transition-transform"
        style={{ transform: `translateX(${distance}px)`, transitionDuration: `${speed}ms` }}
      />
      <div className="absolute right-4 top-4 text-sm text-gray-600">
        <div>Distance: {Math.round(distance)}px</div>
        <div>Speed: {Math.round(speed)}ms</div>
        <div>Effect: {effectLevel}</div>
      </div>
    </div>
  );
}

export default AvatarScene;


