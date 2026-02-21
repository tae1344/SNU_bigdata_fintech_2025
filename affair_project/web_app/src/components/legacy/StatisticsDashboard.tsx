"use client";

import { Card, CardContent } from "@/components/ui/card";

const stats = [
  { label: "국가 데이터", value: "41" },
  { label: "미국 주 데이터", value: "50" },
  { label: "샘플 수", value: "6,368" },
  { label: "분석 모드", value: "실시간" },
];

export function StatisticsDashboard() {
  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
      {stats.map((item) => (
        <Card key={item.label}>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-gray-900">{item.value}</div>
            <div className="text-sm text-gray-600">{item.label}</div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
