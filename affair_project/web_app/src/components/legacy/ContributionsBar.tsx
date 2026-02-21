"use client";

import { Filters } from "@/types";

type Contribution = { key: keyof Filters; value: number };

type ContributionsBarProps = {
  contributions: Contribution[];
};

const labelMap: Record<keyof Filters, string> = {
  rate_marriage: "결혼 만족도",
  age: "나이",
  yrs_married: "결혼 기간",
  children: "자녀 수",
  religious: "종교 성향",
  educ: "교육 수준",
  occupation: "직업",
  occupation_husb: "배우자 직업",
};

export function ContributionsBar({ contributions }: ContributionsBarProps) {
  if (!contributions.length) {
    return <p className="text-sm text-gray-500">표시할 영향 요인이 없습니다.</p>;
  }

  const maxAbs = Math.max(...contributions.map((c) => Math.abs(c.value)), 1);

  return (
    <div className="space-y-3">
      {contributions.map((c) => {
        const widthPercent = (Math.abs(c.value) / maxAbs) * 100;
        const barColor = c.value >= 0 ? "bg-red-500" : "bg-blue-500";
        return (
          <div key={c.key} className="space-y-1">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-700">{labelMap[c.key]}</span>
              <span className="font-medium text-gray-900">{c.value.toFixed(3)}</span>
            </div>
            <div className="h-2 w-full rounded bg-gray-200">
              <div className={`h-2 rounded ${barColor}`} style={{ width: `${widthPercent}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
