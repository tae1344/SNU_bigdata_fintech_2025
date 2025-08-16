"use client";
import React from "react";

type Filters = {
  rate_marriage: number;
  age: number;
  yrs_married: number;
  children: number;
  religious: number;
  educ: number;
  occupation: number;
  occupation_husb: number;
};

type Props = {
  filters: Filters;
};

export function InputValidation({ filters }: Props) {
  const errors: string[] = [];
  if (filters.age < 18 || filters.age > 80) errors.push("나이는 18~80 사이여야 합니다.");
  if (filters.rate_marriage < 1 || filters.rate_marriage > 5) errors.push("결혼 만족도는 1~5입니다.");
  if (filters.religious < 1 || filters.religious > 4) errors.push("종교 성향은 1~4입니다.");
  ["yrs_married", "children", "educ", "occupation", "occupation_husb"].forEach((k) => {
    const v = (filters as any)[k];
    if (v < 0) errors.push(`${k} 값은 0 이상이어야 합니다.`);
  });

  if (errors.length === 0) return null;

  return (
    <ul className="list-inside list-disc rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
      {errors.map((e, i) => (
        <li key={i}>{e}</li>
      ))}
    </ul>
  );
}

export default InputValidation;


