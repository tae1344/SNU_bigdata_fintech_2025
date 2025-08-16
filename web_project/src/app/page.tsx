'use client'
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Shield } from "lucide-react";
import { AnimatePresence } from "motion/react";
import { useEffect, useState } from "react";
import Opening from "../components/Opening";
import OfficeSelection from "../components/OfficeSelection";
import TargetSelection from "../components/TargetSelection";
import InfidelityTest from "../components/InfidelityTest";
import ThemeToggle from "../components/ThemeToggle";
import { useThemeColors } from "../hooks/useThemeColors";
import { useStore } from "@/store/useStore";
import { Loading } from "@/components/Loading";

// 메인 페이지 컴포넌트
export default function MainPage() {
  const [currentStep, setCurrentStep] = useState(0);
  const { colors, isDark } = useThemeColors();

  const {loading, setLoading } = useStore();

  useEffect(() => {
    const timer = setTimeout(() => {  
      setLoading(false);
    }, 2000);

    return () => clearTimeout(timer);
  }, [loading]);

  const steps = [
    "오프닝",
    "사무소 장소 선정", 
    "타겟 선정",
    "개인 진단 테스트",
    "상상의 나래"
  ];

  return (
    <div 
      className="min-h-screen w-full transition-all duration-300"
      style={{
        background: isDark 
          ? `linear-gradient(to bottom, ${colors.background.primary}, ${colors.background.secondary})`
          : `linear-gradient(to bottom, ${colors.background.primary}, ${colors.background.secondary})`
      }}
    >
      <ThemeToggle />
      
      {/* 헤더 */}
      <header 
        className="border-b backdrop-blur sticky top-0 z-10 transition-all duration-300"
        style={{
          borderColor: colors.border,
          backgroundColor: isDark ? 'rgba(15, 23, 42, 0.8)' : 'rgba(239, 246, 255, 0.8)'
        }}
      >
        <div className="max-w-7xl mx-auto px-4 md:px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Shield 
              className="transition-colors duration-300" 
              style={{ color: colors.brand.primary }}
            />
            <h1 
              className="text-xl md:text-2xl font-bold transition-colors duration-300"
              style={{ color: colors.text.primary }}
            >
              엉덩이 탐정 - 불륜 증거 수집 전문 사설 탐정
            </h1>
          </div>
          <div className="hidden md:flex gap-2">
            <Button 
              variant="secondary" 
              className="transition-all duration-300 hover:scale-105"
              style={{
                backgroundColor: colors.background.button,
                color: colors.text.primary,
                border: `1px solid ${colors.border}`
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = colors.background.buttonHover;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = colors.background.button;
              }}
            >
              데이터 분석
            </Button>
            <Button 
              variant="secondary" 
              className="transition-all duration-300 hover:scale-105"
              style={{
                backgroundColor: colors.background.button,
                color: colors.text.primary,
                border: `1px solid ${colors.border}`
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = colors.background.buttonHover;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = colors.background.button;
              }}
            >
              예측 모델
            </Button>
            <Button 
              variant="secondary" 
              className="transition-all duration-300 hover:scale-105"
              style={{
                backgroundColor: colors.background.button,
                color: colors.text.primary,
                border: `1px solid ${colors.border}`
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = colors.background.buttonHover;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = colors.background.button;
              }}
            >
              인사이트
            </Button>
          </div>
        </div>
      </header>

      {loading ? (
          <>
            <Loading />
            <div className="fixed top-20 right-4 bg-red-500 text-white p-2 rounded z-50">
              Loading: {String(loading)}
            </div>
          </>
        ) : (
          <>
            {/* 진행률 표시 */}
            <div 
              className="max-w-7xl mx-auto px-4 md:px-6 py-4 transition-all duration-300"
              role="progressbar" 
              aria-valuenow={currentStep + 1} 
              aria-valuemin={1} 
              aria-valuemax={steps.length}
            >
              <div className="flex items-center gap-4 mb-6">
                <div className="flex-1">
                  <Progress 
                    value={(currentStep / (steps.length - 1)) * 100} 
                    className="h-2 transition-all duration-300"
                    style={{
                      backgroundColor: colors.background.tertiary,
                      color: colors.brand.primary
                    }}
                  />
                </div>
                <span 
                  className="text-sm font-medium transition-colors duration-300"
                  style={{ color: colors.text.primary }}
                >
                  {currentStep + 1} / {steps.length}
                </span>
              </div>
              <div 
                className="flex justify-between text-sm transition-all duration-300"
                style={{ color: colors.text.primary }}
              >
                {steps.map((step, index) => (
                  <span 
                    key={index} 
                    className={`font-medium transition-all duration-300 ${
                      index <= currentStep ? 'opacity-100' : 'opacity-50'
                    }`}
                    aria-current={index === currentStep ? 'step' : undefined}
                    style={{
                      color: index <= currentStep ? colors.text.primary : colors.text.quinary
                    }}
                  >
                    {step}
                  </span>
                ))}
              </div>
            </div>

            {/* 메인 컨텐츠 */}
            <main className="max-w-7xl mx-auto px-4 md:px-6 py-8">
              <AnimatePresence mode="wait">
                {
                  currentStep === 0 && (
                    <Opening nextStep={() => setCurrentStep(1)} />
                  )
                }
                {
                  currentStep === 1 && (
                    <OfficeSelection nextStep={() => setCurrentStep(2)} />
                  )
                }
                {
                  currentStep === 2 && (
                    <TargetSelection nextStep={() => setCurrentStep(3)} />
                  )
                }
                {
                  currentStep === 3 && (
                    <InfidelityTest nextStep={() => setCurrentStep(4)} />
                  )
                }
              </AnimatePresence>
            </main>
          </>
        )}
    </div>
  );
}