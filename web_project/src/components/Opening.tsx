import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Shield, Target, Zap } from "lucide-react";
import { motion } from "motion/react";
import { useThemeColors } from '../hooks/useThemeColors';
import Section from "./Section";
import StatCard from "./StatCard";
import Image from "next/image";

type OpeningProps = {
  nextStep: () => void;
}

export default function Opening({ nextStep }: OpeningProps) {
    const { colors, isDark } = useThemeColors();

    return (
       <motion.div
        key="opening"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.5 }}
      >
        <Section 
          title="🎬 오프닝 – 사설 탐정소 개업 프로젝트"
          subtitle="GSS 데이터 기반: 외노자 인생 5년차의 꿈을 향한 여정"
        >
          <Card 
            className="border-0 shadow-lg transition-all duration-300" 
            style={{ 
              backgroundColor: colors.background.card,
              border: `1px solid ${colors.border}`
            }}
          >
            <CardContent className="p-8 text-center">
              <div className="mb-6">
                <div className="mb-3">
                  <div className="w-36 h-36 mx-auto bg-white rounded-full shadow-lg flex items-center justify-center">
                    <Image src="/images/detective_cat.png" alt="logo" width={80} height={80} className="object-cover" />
                  </div>
                </div>

                <h3 
                  className="text-2xl font-bold mb-4 transition-colors duration-300" 
                  style={{ color: colors.text.primary }}
                >
                  고양이 탐정
                </h3>
                <p 
                  className="text-lg mb-2 transition-colors duration-300" 
                  style={{ color: colors.text.secondary }}
                >
                  불륜 증거 수집 전문 사설 탐정
                </p>
              </div>
              
              <div 
                className="rounded-lg p-6 mb-6 shadow-sm transition-all duration-300" 
                style={{ 
                  backgroundColor: isDark ? colors.background.icon : colors.background.primary,
                  border: `1px solid ${colors.border}`
                }}
              >
                <p 
                  className="text-lg leading-relaxed mb-4 transition-colors duration-300" 
                  style={{ color: colors.text.primary }}
                >
                  &quot;외노자 인생 5년차.. 열심히 차곡차곡 돈을 모았더니 드디어 꿈에 그리던 사설 탐정 사무소를 열 수 있게 되었어..! 
                  그런데,,, 어떻게 사업을 시작해야 할까..? GSS 데이터를 활용해서 데이터 기반으로 접근해보자!&quot;
                </p>
              </div>

              <div className="grid md:grid-cols-3 gap-4 mb-6">
                <StatCard icon={Target} label="목표" value="데이터 기반 전문성" />
                <StatCard icon={Shield} label="핵심 가치" value="GSS 데이터 신뢰성" />
                <StatCard icon={Zap} label="차별화" value="머신러닝 예측" />
              </div>

              <Button 
                onClick={nextStep}
                className="px-8 py-3 text-lg font-semibold transition-all duration-300 hover:scale-105"
                style={{ 
                  backgroundColor: colors.brand.primary,
                  color: '#ffffff'
                }}
              >
                다음 단계로 →
              </Button>
            </CardContent>
          </Card>
        </Section>
      </motion.div>
    )
}