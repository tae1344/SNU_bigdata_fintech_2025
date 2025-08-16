"use client";
import React from "react";

type Props = {
  className?: string;
};

export function EthicsNotice({ className = "" }: Props) {
  return (
    <div className={`rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-800 ${className}`}>
      ⚠️ 모든 결과는 오락/콘텐츠용이며 실제 사실과 무관합니다. 민감 정보는 저장되지 않습니다.
    </div>
  );
}

export default EthicsNotice;


