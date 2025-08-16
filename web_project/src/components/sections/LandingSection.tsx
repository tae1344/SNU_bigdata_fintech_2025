"use client";
import { EthicsNotice } from "@/components/legacy/EthicsNotice";
import { Button } from "@/components/ui/button";
import { useStore } from "@/store/useStore";

export function LandingSection() {
  const { setCurrentStep } = useStore();

  return (
    <div className="flex flex-col items-center justify-center min-h-screen px-4 text-center">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
          불륜 확률 시각화
        </h1>
        
        <p className="text-xl md:text-2xl text-gray-600 mb-8 leading-relaxed">
          데이터 기반 입력으로 가상의 &ldquo;페르소나 캐릭터&rdquo;를 생성하고<br />
          추정 불륜 확률을 게이지와 애니메이션으로 시각화해보세요
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
          <div className="p-6 bg-blue-50 rounded-lg">
            <div className="text-3xl mb-2">📊</div>
            <h3 className="font-semibold mb-2">데이터 기반</h3>
            <p className="text-sm text-gray-600">실제 데이터셋 기반 통계 모델</p>
          </div>
          <div className="p-6 bg-green-50 rounded-lg">
            <div className="text-3xl mb-2">🎭</div>
            <h3 className="font-semibold mb-2">캐릭터 애니메이션</h3>
            <p className="text-sm text-gray-600">확률에 따른 동적 반응</p>
          </div>
          <div className="p-6 bg-purple-50 rounded-lg">
            <div className="text-3xl mb-2">📱</div>
            <h3 className="font-semibold mb-2">공유 가능</h3>
            <p className="text-sm text-gray-600">결과 저장 및 공유</p>
          </div>
        </div>
        
        <Button 
          size="lg" 
          className="text-lg px-8 py-4"
          onClick={() => setCurrentStep("filters")}
        >
          시작하기
        </Button>
        
        <EthicsNotice className="mt-8" />
      </div>
    </div>
  );
}
