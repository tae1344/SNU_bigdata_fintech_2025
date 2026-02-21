"use client";

type AvatarSceneProps = {
  probability: number;
};

export function AvatarScene({ probability }: AvatarSceneProps) {
  const p = Math.max(0, Math.min(1, probability));
  const left = `${p * 70}%`;
  const speedClass = p > 0.66 ? "duration-300" : p > 0.33 ? "duration-500" : "duration-700";

  return (
    <div className="relative h-32 w-full overflow-hidden rounded-lg border border-gray-200 bg-gradient-to-r from-sky-50 to-white">
      <div className="absolute bottom-0 h-2 w-full bg-gray-300" />
      <div className={`absolute bottom-2 text-3xl transition-all ${speedClass}`} style={{ left }}>
        🐱
      </div>
      <div className="absolute left-3 top-3 rounded-md bg-white/80 px-2 py-1 text-xs text-gray-600">
        run: {(p * 100).toFixed(1)}%
      </div>
    </div>
  );
}
