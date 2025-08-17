"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MapPin, Trophy } from "lucide-react";
import { useThemeColors } from "../hooks/useThemeColors";

interface OfficeSelectionAnswerRevealProps {
  className?: string;
}

export default function OfficeSelectionAnswerReveal({ className }: OfficeSelectionAnswerRevealProps) {
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
                  <div className="flex flex-col items-center justify-center gap-4 mb-6">
                    <div className="flex items-center justify-center gap-4">
                      <MapPin 
                        size={48} 
                        style={{ color: colors.brand.success }}
                      />
                      <div className="text-center flex flex-row items-center">
                        <h3 
                          className="text-4xl font-bold transition-colors duration-300"
                          style={{ color: colors.brand.success }}
                        >
                          HAWAII 🌺
                        </h3>
                      </div>
                    </div>
                    <p 
                      className="text-lg transition-colors duration-300"
                      style={{ color: colors.text.secondary }}
                    >
                      1위
                    </p>
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
                            <strong>불륜률:</strong> 80.56% (미국 1위)
                          </span>
                        </li>
                        <li className="flex items-center gap-2">
                          <span className="w-3 h-3 rounded-full" style={{ backgroundColor: colors.brand.success }}></span>
                          <span style={{ color: colors.text.primary }}>
                            <strong>HAWAII:</strong> 140만 명+ 인구, GDP $100B+
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
                            <strong>최고 수요:</strong> 미국 1위 불륜률
                          </span>
                        </li>
                        <li className="flex items-center gap-2">
                          <span className="w-3 h-3 rounded-full" style={{ backgroundColor: colors.brand.warning }}></span>
                          <span style={{ color: colors.text.primary }}>
                            <strong>HAWAII:</strong> 관광지, 휴양지, 다문화 환경
                          </span>
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </CardContent>
    </Card>
  );
}