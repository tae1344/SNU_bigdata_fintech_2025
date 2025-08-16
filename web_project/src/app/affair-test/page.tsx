"use client";
import { AnimatedSection } from "@/components/legacy/AnimatedSection";
import { Loading } from "@/components/Loading";
import { ProgressBar } from "@/components/legacy/ProgressBar";
import { CompareSection } from "@/components/sections/CompareSection";
import { FiltersSection } from "@/components/sections/FiltersSection";
import { LandingSection } from "@/components/sections/LandingSection";
import { MapsSection } from "@/components/sections/MapsSection";
import { ResultSection } from "@/components/sections/ResultSection";
import { useStore } from "@/store/useStore";
import { useEffect } from "react";

export default function TestPage() {
  const { currentStep, loading, setLoading } = useStore();

  useEffect(() => {
    const timer = setTimeout(() => {  
      setLoading(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, [loading]);

  // 단계별 애니메이션 방향 설정
  const getAnimationDirection = (step: string) => {
    switch (step) {
      case "landing":
        return "up";
      case "filters":
        return "right";
      case "result":
        return "right";
      case "compare":
        return "right";
      default:
        return "right";
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* 상단 진행률 바 */}
      <ProgressBar />
      
      {/* 메인 컨텐츠 영역 */}
      <main>
        {loading ? (
          <>
            <Loading />
            <div className="fixed top-20 right-4 bg-red-500 text-white p-2 rounded z-50">
              Loading: {String(loading)}
            </div>
          </>
        ) : (
          <>
            {/* <Image src="/images/giphy_cat.gif" alt="background" width={500} height={500} /> */}
            {/* 동적 컨텐츠 영역 */}
            <div className="min-h-screen">
              <AnimatedSection 
                isVisible={currentStep === "landing"} 
                direction="up"
                className="w-full"
              >
                <LandingSection />
              </AnimatedSection>
              
              <AnimatedSection 
                isVisible={currentStep === "filters"} 
                direction="right"
                className="w-full"
              >
                <FiltersSection />
              </AnimatedSection>
              
              <AnimatedSection 
                isVisible={currentStep === "result"} 
                direction="right"
                className="w-full"
              >
                <ResultSection />
              </AnimatedSection>
              
              <AnimatedSection 
                isVisible={currentStep === "compare"} 
                direction="right"
                className="w-full"
              >
                <CompareSection />
              </AnimatedSection>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
