"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MapPin, Trophy } from "lucide-react";
import { useThemeColors } from "../hooks/useThemeColors";

interface AnswerRevealProps {
  className?: string;
}

export default function AnswerReveal({ className }: AnswerRevealProps) {
  const { colors, isDark } = useThemeColors();
  const [showAnswer, setShowAnswer] = useState(false);

  return (
    <Card 
      className={`mb-8 shadow-lg transition-all duration-300 ${className || ''}`}
      style={{
        backgroundColor: colors.background.card,
        border: `1px solid ${colors.border}`
      }}
    >
      <CardHeader className="text-center">
        <div className="flex items-center justify-center gap-3 mb-4">
          <Trophy 
            size={32} 
            style={{ color: colors.brand.warning }}
            className="animate-bounce"
          />
          <CardTitle 
            className="text-3xl font-bold transition-colors duration-300"
            style={{ color: colors.text.primary }}
          >
            정답 공개!
          </CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        <div className="text-center">
          <Button
            onClick={() => setShowAnswer(!showAnswer)}
            className="mb-6 px-8 py-4 text-lg font-semibold transition-all duration-300 hover:scale-105"
            style={{
              backgroundColor: colors.brand.success,
              color: '#ffffff'
            }}
          >
            {showAnswer ? '정답 숨기기' : '정답 보기'}
          </Button>
          
          <AnimatePresence>
            {showAnswer && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                transition={{ duration: 0.5 }}
                className="space-y-6"
              >
                <div 
                  className="p-8 rounded-lg transition-all duration-300"
                  style={{
                    backgroundColor: isDark ? colors.background.icon : colors.background.primary,
                    border: `2px solid ${colors.brand.success}`
                  }}
                >
                  <div className="flex items-center justify-center gap-4 mb-6">
                    <MapPin 
                      size={48} 
                      style={{ color: colors.brand.success }}
                    />
                    <h3 
                      className="text-4xl font-bold transition-colors duration-300"
                      style={{ color: colors.brand.success }}
                    >
                      THAILAND 🇹🇭
                    </h3>
                  </div>
                  
                  <div className="grid md:grid-cols-2 gap-6 text-left">
                    <div>
                      <h4 
                        className="text-xl font-semibold mb-3 transition-colors duration-300"
                        style={{ color: colors.text.primary }}
                      >
                        📊 데이터 분석 결과
                      </h4>
                      <ul className="space-y-2">
                        <li className="flex items-center gap-2">
                          <span className="w-3 h-3 rounded-full" style={{ backgroundColor: colors.brand.success }}></span>
                          <span style={{ color: colors.text.primary }}>
                            <strong>바람지수:</strong> 50% (세계 1위)
                          </span>
                        </li>
                        <li className="flex items-center gap-2">
                          <span className="w-3 h-3 rounded-full" style={{ backgroundColor: colors.brand.success }}></span>
                          <span style={{ color: colors.text.primary }}>
                            <strong>관광객 유입:</strong> 연간 4,000만 명+
                          </span>
                        </li>
                        <li className="flex items-center gap-2">
                          <span className="w-3 h-3 rounded-full" style={{ backgroundColor: colors.brand.success }}></span>
                          <span style={{ color: colors.text.primary }}>
                            <strong>경제 규모:</strong> GDP $500B+
                          </span>
                        </li>
                      </ul>
                    </div>
                    
                    <div>
                      <h4 
                        className="text-xl font-semibold mb-3 transition-colors duration-300"
                        style={{ color: colors.text.primary }}
                      >
                        💡 사업 기회 요인
                      </h4>
                      <ul className="space-y-2">
                        <li className="flex items-center gap-2">
                          <span className="w-3 h-3 rounded-full" style={{ backgroundColor: colors.brand.warning }}></span>
                          <span style={{ color: colors.text.primary }}>
                            <strong>최고 수요:</strong> 세계 1위 바람지수
                          </span>
                        </li>
                        <li className="flex items-center gap-2">
                          <span className="w-3 h-3 rounded-full" style={{ backgroundColor: colors.brand.warning }}></span>
                          <span style={{ color: colors.text.primary }}>
                            <strong>관광산업 메카:</strong> 다양한 고객층
                          </span>
                        </li>
                        <li className="flex items-center gap-2">
                          <span className="w-3 h-3 rounded-full" style={{ backgroundColor: colors.brand.warning }}></span>
                          <span style={{ color: colors.text.primary }}>
                            <strong>시장 진입 용이:</strong> 낮은 경쟁 환경
                          </span>
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
                
                <div 
                  className="p-6 rounded-lg transition-all duration-300"
                  style={{
                    backgroundColor: isDark ? colors.background.icon : colors.background.primary,
                    border: `1px solid ${colors.border}`
                  }}
                >
                  <p 
                    className="text-lg leading-relaxed transition-colors duration-300"
                    style={{ color: colors.text.primary }}
                  >
                    🎯 <strong>결론:</strong> 태국은 세계 1위 바람지수(50%)와 연간 4,000만 명의 관광객 유입, 
                    강력한 관광산업을 고려할 때 탐정 사무소 개업에 최적의 지역입니다. 
                    특히 방콕, 푸켓, 치앙마이 등 주요 도시에서 관광객과 현지인을 대상으로 한 
                    높은 수요와 낮은 경쟁 환경으로 시장 진입이 용이하며, 
                    다국적 고객층을 대상으로 한 프리미엄 서비스 제공이 가능합니다.
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </CardContent>
    </Card>
  );
}
