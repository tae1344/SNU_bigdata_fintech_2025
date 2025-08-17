'use client'
import FranchiseSelection from "@/components/FranchiseSelection";
import { Loading } from "@/components/Loading";
import { Progress } from "@/components/ui/progress";
import { useStore } from "@/store/useStore";
import { Shield } from "lucide-react";
import { AnimatePresence } from "motion/react";
import { useEffect, useState } from "react";
import InfidelityTest from "../components/InfidelityTest";
import OfficeSelection from "../components/OfficeSelection";
import Opening from "../components/Opening";
import TargetSelection from "../components/TargetSelection";
import { useThemeColors } from "../hooks/useThemeColors";

// 메인 페이지 컴포넌트
export default function MainPage() {
  const [currentStep, setCurrentStep] = useState(0);
  const { colors, isDark } = useThemeColors();

  const {loading, setLoading } = useStore();

  // 스텝 변경 시 스크롤을 상단으로 이동
  const handleStepChange = (newStep: number) => {
    setCurrentStep(newStep);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

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
      {/* <ThemeToggle /> */}
      
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
              고양이 탐정 - 불륜 증거 수집 전문 사설 탐정
            </h1>
          </div>
        </div>
      </header>

      {loading ? (
          <>
            <Loading />
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
                  <button
                    key={index} 
                    onClick={() => handleStepChange(index)}
                    className={`font-medium transition-all duration-300 cursor-pointer hover:scale-105 hover:underline ${
                      index <= currentStep ? 'opacity-100' : 'opacity-50'
                    } ${
                      index === currentStep ? 'ring-1 ring-opacity-5 ring-white' : ''
                    }`}
                    aria-current={index === currentStep ? 'step' : undefined}
                    style={{
                      color: index <= currentStep ? colors.text.primary : colors.text.quinary,
                      backgroundColor: 'transparent',
                      border: 'none',
                      padding: '4px 8px',
                      borderRadius: '6px'
                    }}
                    disabled={index > currentStep}
                  >
                    {step}
                  </button>
                ))}
              </div>
            </div>

            {/* 메인 컨텐츠 */}
            <main className="max-w-7xl mx-auto px-4 md:px-6 py-8">
              <AnimatePresence mode="wait">
                {
                  currentStep === 0 && (
                    <Opening nextStep={() => handleStepChange(1)} />
                  )
                }
                {
                  currentStep === 1 && (
                    <OfficeSelection nextStep={() => handleStepChange(2)} />
                  )
                }
                {
                  currentStep === 2 && (
                    <TargetSelection nextStep={() => handleStepChange(3)} />
                  )
                }
                {
                  currentStep === 3 && (
                    <InfidelityTest nextStep={() => handleStepChange(4)} />
                  )
                }
                {
                  currentStep === 4 && (
                    <FranchiseSelection nextStep={() => handleStepChange(5)} />
                  )
                }
              </AnimatePresence>
            </main>
          </>
        )}
    </div>
  );
}