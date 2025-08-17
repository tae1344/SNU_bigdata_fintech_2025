"use client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Lightbulb } from "lucide-react";
import { motion } from "motion/react";
import { useState } from "react";
import { useThemeColors } from "../hooks/useThemeColors";
import FranchiseAnswerReveal from "./FranchiseAnswerReveal";
import { WorldInfidelityMap } from "./WorldInfidelityMap";

type FranchiseSelectionProps = {
  nextStep: () => void;
}

export default function FranchiseSelection({ nextStep }: FranchiseSelectionProps) {
  const { colors, isDark } = useThemeColors();
  const [showWorldRankings, setShowWorldRankings] = useState(false);

  return (
    <motion.div
      key="franchise-selection"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.5 }}
      className="min-h-screen transition-all duration-300"
      style={{
        background: isDark 
          ? `linear-gradient(to bottom, ${colors.background.primary}, ${colors.background.secondary})`
          : `linear-gradient(to bottom, ${colors.background.primary}, ${colors.background.secondary})`
      }}
    >
      <div className="container mx-auto px-4 py-8">
        {/* 퀴즈 섹션 */}
        <Card 
          className="mb-8 shadow-lg transition-all duration-300"
          style={{
            backgroundColor: colors.background.card,
            border: `1px solid ${colors.border}`
          }}
        >
          <CardHeader className="text-center">
            <div className="flex items-center justify-center gap-3 mb-4">
              <Lightbulb 
                size={32} 
                style={{ color: colors.brand.warning }}
                className="animate-pulse"
              />
              <CardTitle 
                className="text-3xl font-bold transition-colors duration-300"
                style={{ color: colors.text.primary }}
              >
                🕵️‍♂️ 프랜차이즈 탐정 사무소 위치 선정 퀴즈
              </CardTitle>
            </div>
            <p 
              className="text-lg leading-relaxed transition-colors duration-300"
              style={{ color: colors.text.secondary }}
            >
              <strong>문제:</strong> GSS 데이터와 세계 바람지수를 분석한 결과, 
              <br />
              <span className="text-xl font-semibold" style={{ color: colors.brand.danger }}>
                &ldquo;어느 나라에 프랜차이즈 탐정 사무소를 차리면 가장 수익성이 좋을까요?&rdquo;
              </span>
            </p>
            <div 
              className="mt-4 p-4 rounded-lg transition-all duration-300"
              style={{
                backgroundColor: isDark ? colors.background.icon : colors.background.primary,
                border: `1px solid ${colors.border}`
              }}
            >
              <p 
                className="text-base transition-colors duration-300"
                style={{ color: colors.text.primary }}
              >
                💡 <strong>힌트:</strong> 지도를 자세히 살펴보고, 각 나라의 통계를 분석해보세요!
                <br />
                📊 데이터를 토글하여 자세한 정보를 확인할 수 있습니다.
              </p>
            </div>
          </CardHeader>
        </Card>

        {/* 세계 지도 */}
        <Card 
          className="mb-8 shadow-lg transition-all duration-300"
          style={{
            backgroundColor: colors.background.card,
            border: `1px solid ${colors.border}`
          }}
        >
          <CardHeader>
            <CardTitle 
              className="text-2xl font-bold transition-colors duration-300"
              style={{ color: colors.text.primary }}
            >
              🌍 세계 국가별 바람지수 2025
            </CardTitle>
            <p 
              className="transition-colors duration-300"
              style={{ color: colors.text.quaternary }}
            >
              나라별 바람 지수(자기보고형 설문 %). 파란색이 진할수록 비율이 높습니다.
            </p>
          </CardHeader>
          <CardContent>
            <WorldInfidelityMap 
              showRankings={showWorldRankings}
              onToggleRankings={() => setShowWorldRankings(!showWorldRankings)}
            />
          </CardContent>
        </Card>

        {/* 정답 공개 섹션 */}
        <FranchiseAnswerReveal />

        {/* 다음 단계 버튼 */}
        <div className="text-center">
          <Button 
            onClick={nextStep}
            className="px-8 py-4 text-lg font-semibold transition-all duration-300 hover:scale-105"
            style={{
              backgroundColor: colors.brand.primary,
              color: '#ffffff'
            }}
          >
            다음 단계로 →
          </Button>
        </div>
      </div>
    </motion.div>
  );
}
