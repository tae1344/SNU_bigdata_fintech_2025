"use client";

type EthicsNoticeProps = {
  className?: string;
};

export function EthicsNotice({ className = "" }: EthicsNoticeProps) {
  return (
    <div className={`rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900 ${className}`}>
      <strong>윤리 고지:</strong> 본 결과는 오락/학습용 시뮬레이션이며, 실제 개인의 행동을 진단하거나
      예측하는 용도로 사용할 수 없습니다.
    </div>
  );
}
