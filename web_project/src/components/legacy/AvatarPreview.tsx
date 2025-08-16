"use client";
import React from "react";

type Props = {
  age: number;
  children: number;
  occupation: number;
};

export function AvatarPreview({ age, children, occupation }: Props) {
  // 간단한 외형 매핑
  const hair = age < 30 ? "bg-yellow-400" : age < 50 ? "bg-amber-800" : "bg-gray-400";
  const accessory = children > 0 ? "👶" : "";
  const suit = occupation >= 4 ? "bg-blue-600" : "bg-emerald-600";

  return (
    <div className="flex items-center gap-4">
      <div className="relative h-24 w-24">
        <div className={`absolute left-1/2 top-1/2 h-24 w-24 -translate-x-1/2 -translate-y-1/2 rounded-full bg-pink-200`} />
        <div className={`absolute left-1/2 top-2 h-8 w-16 -translate-x-1/2 rounded ${hair}`} />
        <div className={`absolute bottom-0 left-1/2 h-10 w-20 -translate-x-1/2 rounded-b ${suit}`} />
        <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 text-2xl">{accessory}</div>
      </div>
      <div className="text-sm text-gray-700">
        <div>Age: {age}</div>
        <div>Children: {children}</div>
        <div>Occupation: {occupation}</div>
      </div>
    </div>
  );
}

export default AvatarPreview;


