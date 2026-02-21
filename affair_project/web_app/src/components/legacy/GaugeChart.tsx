"use client";

type GaugeChartProps = {
  value: number;
  label?: string;
};

export function GaugeChart({ value, label = "Score" }: GaugeChartProps) {
  const safeValue = Math.max(0, Math.min(100, value));
  const ringColor =
    safeValue < 40 ? "text-blue-600" : safeValue < 70 ? "text-orange-500" : "text-red-600";

  return (
    <div className="relative flex h-36 w-36 items-center justify-center rounded-full border-8 border-gray-200 bg-white">
      <div className="absolute inset-0 rounded-full">
        <div
          className="h-full w-full rounded-full"
          style={{
            background: `conic-gradient(rgb(37 99 235) ${safeValue * 3.6}deg, rgb(229 231 235) 0deg)`,
          }}
        />
      </div>
      <div className="absolute flex h-24 w-24 flex-col items-center justify-center rounded-full bg-white shadow-sm">
        <span className={`text-2xl font-bold ${ringColor}`}>{safeValue}</span>
        <span className="text-xs text-gray-500">{label}</span>
      </div>
    </div>
  );
}
