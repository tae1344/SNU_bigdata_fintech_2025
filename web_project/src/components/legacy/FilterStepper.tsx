"use client";
import React, { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb"


export type Filters = {
  rate_marriage: 1 | 2 | 3 | 4 | 5;
  age: number;
  yrs_married: number;
  children: number;
  religious: 1 | 2 | 3 | 4;
  educ: number;
  occupation: number;
  occupation_husb: number;
};

type Props = {
  initial?: Partial<Filters>;
  onChange?: (filters: Filters) => void;
  onComplete?: (filters: Filters) => void;
};

const DEFAULTS: Filters = {
  rate_marriage: 3,
  age: 35,
  yrs_married: 8,
  children: 1,
  religious: 2,
  educ: 3,
  occupation: 2,
  occupation_husb: 2,
};

export function FilterStepper({ initial, onChange, onComplete }: Props) {
  const [step, setStep] = useState(0);
  const [filters, setFilters] = useState<Filters>({
    ...DEFAULTS,
    ...(initial ?? {}),
  });

  const steps = useMemo(
    () => [
      {
        title: "기본 정보",
        fields: [
          { key: "rate_marriage", label: "결혼 만족도 (1-5)", min: 1, max: 5 },
          { key: "age", label: "나이", min: 18, max: 80 },
          { key: "yrs_married", label: "결혼 기간(년)", min: 0, max: 60 },
        ] as const,
      },
      {
        title: "가정/가치관",
        fields: [
          { key: "children", label: "자녀 수", min: 0, max: 10 },
          { key: "religious", label: "종교 성향 (1-4)", min: 1, max: 4 },
        ] as const,
      },
      {
        title: "교육/직업",
        fields: [
          { key: "educ", label: "교육 수준 코드", min: 1, max: 6 },
          { key: "occupation", label: "직업 코드", min: 1, max: 6 },
          { key: "occupation_husb", label: "배우자 직업 코드", min: 1, max: 6 },
        ] as const,
      },
    ],
    []
  );

  function update<K extends keyof Filters>(key: K, value: number) {
    const next: Filters = { ...filters, [key]: value } as Filters;
    setFilters(next);
    onChange?.(next);
  }

  function nextStep() {
    if (step < steps.length - 1) setStep((s) => s + 1);
    else onComplete?.(filters);
  }

  function prevStep() {
    if (step > 0) setStep((s) => s - 1);
  }

  const current = steps[step];

  return (
    <div className="w-full max-w-2xl rounded-lg border bg-white p-6 shadow">
      <div className="mb-4 text-sm text-gray-500">Step {step + 1} / {steps.length}</div>
      <h2 className="mb-6 text-2xl font-semibold">{current.title}</h2>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {current.fields.map((f) => (
          <label key={String(f.key)} className="flex flex-col gap-2">
            <span className="text-sm text-gray-700">{f.label}</span>
            <input
              type="number"
              className="rounded border px-3 py-2 focus:outline-none focus:ring"
              min={f.min}
              max={f.max}
              value={Number(filters[f.key])}
              onChange={(e) => update(f.key, Number(e.target.value))}
            />
          </label>
        ))}
      </div>

      <div className="mt-6 flex items-center justify-between">
        <Button
          onClick={prevStep}
          disabled={step === 0}
        >
          이전
        </Button>
        <Button
          onClick={nextStep}
        >
          {step === steps.length - 1 ? "완료" : "다음"}
        </Button>
      </div>
    </div>
  );
}

export default FilterStepper;


