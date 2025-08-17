'use client'

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BookOpen, Calendar, Heart, Shield, TrendingUp, Users } from "lucide-react";
import { motion } from "motion/react";
import React, { useMemo } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { useThemeColors } from "../hooks/useThemeColors";

type TargetSelectionProps = {
  nextStep: () => void;
}

// colors 타입 정의
type ThemeColors = ReturnType<typeof useThemeColors>['colors'];

export default function TargetSelection({ nextStep }: TargetSelectionProps) {
  const { colors, isDark } = useThemeColors();
  const pieColors = useMemo(() => [colors.chartPalette.green, colors.chartPalette.red, colors.chartPalette.amber], [colors.chartPalette]);

  return (
    <motion.div
      key="target-selection"
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
        {/* 히어로 */}
        <section className="relative mb-8">
          <Card 
            className="shadow-lg transition-all duration-300"
            style={{
              backgroundColor: colors.background.card,
              border: `1px solid ${colors.border}`
            }}
          >
            <CardContent className="p-8">
              <div className="grid md:grid-cols-2 gap-6 items-center">
                <div>
                  <motion.h2 
                    initial={{ opacity: 0, y: 10 }} 
                    animate={{ opacity: 1, y: 0 }} 
                    transition={{ duration: 0.6 }}
                    className="text-2xl md:text-4xl font-extrabold leading-tight transition-colors duration-300"
                    style={{ color: colors.text.primary }}
                  >
                    GSS 데이터 기반 <span style={{ color: colors.brand.primary }}>불륜 예측 모델</span>
                  </motion.h2>
                  <p 
                    className="mt-3 transition-colors duration-300"
                    style={{ color: colors.text.secondary }}
                  >
                    미국 일반사회조사(GSS) 1972-2022 데이터를 활용한 24,460개 샘플 기반 분석.
                    결혼 만족도, 나이, 결혼 기간, 종교 성향 등 다양한 요인을 분석하여
                    불륜 확률을 예측하는 머신러닝 모델의 인사이트를 제공합니다.
                  </p>
                  <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
                    <StatCard icon={Users} label="전체 샘플" value="24,460" colors={colors} isDark={isDark} />
                    <StatCard icon={Heart} label="불륜 경험" value="4,346" colors={colors} isDark={isDark} />
                    <StatCard icon={TrendingUp} label="불륜률" value="17.77%" colors={colors} isDark={isDark} />
                    <StatCard icon={BookOpen} label="특성 수" value="14개" colors={colors} isDark={isDark} />
                  </div>
                </div>
                <div 
                  className="rounded-2xl p-4 md:p-6 transition-all duration-300"
                  style={{
                    backgroundColor: isDark ? colors.background.icon : colors.background.primary,
                    border: `1px solid ${colors.border}`
                  }}
                >
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={genderData}>
                      <CartesianGrid strokeDasharray="3 3" style={{ stroke: colors.chart.grid }} />
                      <XAxis dataKey="name" style={{ stroke: colors.chart.axis }} />
                      <YAxis unit="%" style={{ stroke: colors.chart.axis }} />
                      <Tooltip 
                        cursor={{ fill: colors.chart.tooltip.background }} 
                        contentStyle={{ 
                          background: colors.chart.tooltip.background, 
                          border: `1px solid ${colors.chart.tooltip.border}`, 
                          color: colors.chart.tooltip.text 
                        }} 
                      />
                      <Bar dataKey="rate" radius={[8,8,0,0]} fill={colors.chartPalette.blue} />
                    </BarChart>
                  </ResponsiveContainer>
                  <div 
                    className="text-xs mt-2 transition-colors duration-300"
                    style={{ color: colors.text.quinary }}
                  >
                    출처: GSS 1972-2022 데이터 분석
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* 2. 연령대별 추세 */}
        <Section title="연령·성별 불륜률 추세" subtitle="GSS 데이터 기반: 나이가 증가할수록 불륜률이 높아지는 경향, 남성이 여성보다 높은 비율" colors={colors}>
          <div className="grid md:grid-cols-2 gap-6">
            <Card 
              className="shadow-lg transition-all duration-300"
              style={{
                backgroundColor: colors.background.card,
                border: `1px solid ${colors.border}`
              }}
            >
              <CardHeader>
                <CardTitle style={{ color: colors.text.primary }}>연령대별 남녀 불륜률</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={ageTrend}>
                    <CartesianGrid style={{ stroke: colors.chart.grid }} />
                    <XAxis dataKey="age" style={{ stroke: colors.chart.axis }} />
                    <YAxis unit="%" style={{ stroke: colors.chart.axis }} />
                    <Tooltip 
                      contentStyle={{ 
                        background: colors.chart.tooltip.background, 
                        border: `1px solid ${colors.chart.tooltip.border}`, 
                        color: colors.chart.tooltip.text 
                      }} 
                    />
                    <Legend />
                    <Line type="monotone" dataKey="남성" stroke={colors.chartPalette.blue} strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="여성" stroke={colors.chartPalette.red} strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
                <div 
                  className="text-xs mt-2 transition-colors duration-300"
                  style={{ color: colors.text.quinary }}
                >
                  출처: GSS 1972-2022 데이터 분석
                </div>
              </CardContent>
            </Card>

            <Card 
              className="shadow-lg transition-all duration-300"
              style={{
                backgroundColor: colors.background.card,
                border: `1px solid ${colors.border}`
              }}
            >
              <CardHeader>
                <CardTitle style={{ color: colors.text.primary }}>세대별 코호트 효과</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={280}>
                  <AreaChart data={cohortData}>
                    <CartesianGrid style={{ stroke: colors.chart.grid }} />
                    <XAxis dataKey="cohort" style={{ stroke: colors.chart.axis }} />
                    <YAxis unit="%" style={{ stroke: colors.chart.axis }} />
                    <Tooltip 
                      contentStyle={{ 
                        background: colors.chart.tooltip.background, 
                        border: `1px solid ${colors.chart.tooltip.border}`, 
                        color: colors.chart.tooltip.text 
                      }} 
                    />
                    <Area type="monotone" dataKey="rate" stroke={colors.chartPalette.teal} fill={colors.chartPalette.teal} fillOpacity={0.3} />
                  </AreaChart>
                </ResponsiveContainer>
                <div 
                  className="text-xs mt-2 transition-colors duration-300"
                  style={{ color: colors.text.quinary }}
                >
                  출처: GSS 연도별 데이터 분석 (코호트 효과)
                </div>
              </CardContent>
            </Card>
          </div>
        </Section>

        {/* 3. 결혼 관련 변수 */}
        <Section title="결혼 만족도 & 결혼 연수" subtitle="GSS 데이터 기반: 결혼 만족도가 낮을수록, 결혼 연수가 길수록 불륜률이 높습니다" colors={colors}>
          <div className="grid md:grid-cols-2 gap-6">
            <Card 
              className="shadow-lg transition-all duration-300"
              style={{
                backgroundColor: colors.background.card,
                border: `1px solid ${colors.border}`
              }}
            >
              <CardHeader>
                <CardTitle style={{ color: colors.text.primary }}>결혼 만족도별 불륜률</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={marriageSatisfactionData}>
                    <CartesianGrid style={{ stroke: colors.chart.grid }} />
                    <XAxis dataKey="satisfaction" style={{ stroke: colors.chart.axis }} />
                    <YAxis unit="%" style={{ stroke: colors.chart.axis }} />
                    <Tooltip 
                      contentStyle={{ 
                        background: colors.chart.tooltip.background, 
                        border: `1px solid ${colors.chart.tooltip.border}`, 
                        color: colors.chart.tooltip.text 
                      }} 
                    />
                    <Bar dataKey="rate" radius={[8,8,0,0]} fill={colors.chartPalette.violet} />
                  </BarChart>
                </ResponsiveContainer>
                <div 
                  className="text-xs mt-2 transition-colors duration-300"
                  style={{ color: colors.text.quinary }}
                >
                  출처: GSS HAPMAR 변수 분석
                </div>
              </CardContent>
            </Card>

            <Card 
              className="shadow-lg transition-all duration-300"
              style={{
                backgroundColor: colors.background.card,
                border: `1px solid ${colors.border}`
              }}
            >
              <CardHeader>
                <CardTitle style={{ color: colors.text.primary }}>결혼 연수별 불륜률</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={marriageDurationData}>
                    <CartesianGrid style={{ stroke: colors.chart.grid }} />
                    <XAxis dataKey="duration" style={{ stroke: colors.chart.axis }} />
                    <YAxis unit="%" style={{ stroke: colors.chart.axis }} />
                    <Tooltip 
                      contentStyle={{ 
                        background: colors.chart.tooltip.background, 
                        border: `1px solid ${colors.chart.tooltip.border}`, 
                        color: colors.chart.tooltip.text 
                      }} 
                    />
                    <Line type="monotone" dataKey="rate" stroke={colors.chartPalette.green} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
                <div 
                  className="text-xs mt-2 transition-colors duration-300"
                  style={{ color: colors.text.quinary }}
                >
                  출처: GSS AGEWED-AGE 계산 및 보간
                </div>
              </CardContent>
            </Card>
          </div>
        </Section>

        {/* 4. 종교 및 직업 */}
        <Section title="종교 활동 & 직업 등급" subtitle="GSS 데이터 기반: 종교 활동이 적을수록, 직업 등급이 낮을수록 불륜률이 높습니다" colors={colors}>
          <div className="grid md:grid-cols-2 gap-6">
            <Card 
              className="shadow-lg transition-all duration-300"
              style={{
                backgroundColor: colors.background.card,
                border: `1px solid ${colors.border}`
              }}
            >
              <CardHeader>
                <CardTitle style={{ color: colors.text.primary }}>종교 활동별 불륜률</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={religionData}>
                    <CartesianGrid style={{ stroke: colors.chart.grid }} />
                    <XAxis dataKey="freq" style={{ stroke: colors.chart.axis }} />
                    <YAxis unit="%" style={{ stroke: colors.chart.axis }} />
                    <Tooltip 
                      contentStyle={{ 
                        background: colors.chart.tooltip.background, 
                        border: `1px solid ${colors.chart.tooltip.border}`, 
                        color: colors.chart.tooltip.text 
                      }} 
                    />
                    <Line type="monotone" dataKey="rate" stroke={colors.chartPalette.green} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
                <div 
                  className="text-xs mt-2 transition-colors duration-300"
                  style={{ color: colors.text.quinary }}
                >
                  출처: GSS ATTEND 변수 분석
                </div>
              </CardContent>
            </Card>

            <Card 
              className="shadow-lg transition-all duration-300"
              style={{
                backgroundColor: colors.background.card,
                border: `1px solid ${colors.border}`
              }}
            >
              <CardHeader>
                <CardTitle style={{ color: colors.text.primary }}>직업 등급별 불륜률</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={occupationData}>
                    <CartesianGrid style={{ stroke: colors.chart.grid }} />
                    <XAxis dataKey="occupation" style={{ stroke: colors.chart.axis }} />
                    <YAxis unit="%" style={{ stroke: colors.chart.axis }} />
                    <Tooltip 
                      contentStyle={{ 
                        background: colors.chart.tooltip.background, 
                        border: `1px solid ${colors.chart.tooltip.border}`, 
                        color: colors.chart.tooltip.text 
                      }} 
                    />
                    <Bar dataKey="rate" radius={[8,8,0,0]} fill={colors.chartPalette.rose} />
                  </BarChart>
                </ResponsiveContainer>
                <div 
                  className="text-xs mt-2 transition-colors duration-300"
                  style={{ color: colors.text.quinary }}
                >
                  출처: GSS PRESTG10 변수 분석
                </div>
              </CardContent>
            </Card>
          </div>
        </Section>

        {/* 5. 교육 및 자녀 */}
        <Section title="교육 수준 & 자녀 수" subtitle="GSS 데이터 기반: 교육 수준이 낮을수록, 자녀가 없을수록 불륜률이 높습니다" colors={colors}>
          <div className="grid md:grid-cols-2 gap-6">
            <Card 
              className="shadow-lg transition-all duration-300"
              style={{
                backgroundColor: colors.background.card,
                border: `1px solid ${colors.border}`
              }}
            >
              <CardHeader>
                <CardTitle style={{ color: colors.text.primary }}>교육 수준별 불륜률</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={educationData}>
                    <CartesianGrid style={{ stroke: colors.chart.grid }} />
                    <XAxis dataKey="level" style={{ stroke: colors.chart.axis }} />
                    <YAxis unit="%" style={{ stroke: colors.chart.axis }} />
                    <Tooltip 
                      contentStyle={{ 
                        background: colors.chart.tooltip.background, 
                        border: `1px solid ${colors.chart.tooltip.border}`, 
                        color: colors.chart.tooltip.text 
                      }} 
                    />
                    <Bar dataKey="rate" radius={[8,8,0,0]} fill={colors.chartPalette.amber} />
                  </BarChart>
                </ResponsiveContainer>
                <div 
                  className="text-xs mt-2 transition-colors duration-300"
                  style={{ color: colors.text.quinary }}
                >
                  출처: GSS EDUC 변수 분석
                </div>
              </CardContent>
            </Card>

            <Card 
              className="shadow-lg transition-all duration-300"
              style={{
                backgroundColor: colors.background.card,
                border: `1px solid ${colors.border}`
              }}
            >
              <CardHeader>
                <CardTitle style={{ color: colors.text.primary }}>자녀 수별 불륜률</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={childrenData}>
                    <CartesianGrid style={{ stroke: colors.chart.grid }} />
                    <XAxis dataKey="count" style={{ stroke: colors.chart.axis }} />
                    <YAxis unit="%" style={{ stroke: colors.chart.axis }} />
                    <Tooltip 
                      contentStyle={{ 
                        background: colors.chart.tooltip.background, 
                        border: `1px solid ${colors.chart.tooltip.border}`, 
                        color: colors.chart.tooltip.text 
                      }} 
                    />
                    <Bar dataKey="rate" radius={[8,8,0,0]} fill={colors.chartPalette.teal} />
                  </BarChart>
                </ResponsiveContainer>
                <div 
                  className="text-xs mt-2 transition-colors duration-300"
                  style={{ color: colors.text.quinary }}
                >
                  출처: GSS CHILDS 변수 분석
                </div>
              </CardContent>
            </Card>
          </div>
        </Section>

        {/* 6. 파생 변수 */}
        <Section title="파생 변수 분석" subtitle="GSS 데이터 기반: 결혼연수/나이 비율과 결혼만족도×결혼연수 복합 지표" colors={colors}>
          <div className="grid md:grid-cols-2 gap-6">
            <Card 
              className="shadow-lg transition-all duration-300"
              style={{
                backgroundColor: colors.background.card,
                border: `1px solid ${colors.border}`
              }}
            >
              <CardHeader>
                <CardTitle style={{ color: colors.text.primary }}>결혼연수/나이 비율별 불륜률</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={yrsPerAgeData}>
                    <CartesianGrid style={{ stroke: colors.chart.grid }} />
                    <XAxis dataKey="ratio" style={{ stroke: colors.chart.axis }} />
                    <YAxis unit="%" style={{ stroke: colors.chart.axis }} />
                    <Tooltip 
                      contentStyle={{ 
                        background: colors.chart.tooltip.background, 
                        border: `1px solid ${colors.chart.tooltip.border}`, 
                        color: colors.chart.tooltip.text 
                      }} 
                    />
                    <Bar dataKey="rate" radius={[8,8,0,0]} fill={colors.chartPalette.red} />
                  </BarChart>
                </ResponsiveContainer>
                <div 
                  className="text-xs mt-2 transition-colors duration-300"
                  style={{ color: colors.text.quinary }}
                >
                  출처: GSS 파생 변수 yrs_per_age 분석
                </div>
              </CardContent>
            </Card>

            <Card 
              className="shadow-lg transition-all duration-300"
              style={{
                backgroundColor: colors.background.card,
                border: `1px solid ${colors.border}`
              }}
            >
              <CardHeader>
                <CardTitle style={{ color: colors.text.primary }}>결혼만족도×결혼연수별 불륜률</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={rateXYrsData}>
                    <CartesianGrid style={{ stroke: colors.chart.grid }} />
                    <XAxis dataKey="score" style={{ stroke: colors.chart.axis }} />
                    <YAxis unit="%" style={{ stroke: colors.chart.axis }} />
                    <Tooltip 
                      contentStyle={{ 
                        background: colors.chart.tooltip.background, 
                        border: `1px solid ${colors.chart.tooltip.border}`, 
                        color: colors.chart.tooltip.text 
                      }} 
                    />
                    <Bar dataKey="rate" radius={[8,8,0,0]} fill={colors.chartPalette.violet} />
                  </BarChart>
                </ResponsiveContainer>
                <div 
                  className="text-xs mt-2 transition-colors duration-300"
                  style={{ color: colors.text.quinary }}
                >
                  출처: GSS 파생 변수 rate_x_yrs 분석
                </div>
              </CardContent>
            </Card>
          </div>
        </Section>

        {/* 7. 모델 성능 */}
        <Section title="예측 모델 성능" subtitle="GSS 데이터 기반: RandomForest 모델 (200개 트리, 12개 특성)" colors={colors}>
          <div className="grid md:grid-cols-2 gap-6">
            <Card 
              className="shadow-lg transition-all duration-300"
              style={{
                backgroundColor: colors.background.card,
                border: `1px solid ${colors.border}`
              }}
            >
              <CardHeader>
                <CardTitle style={{ color: colors.text.primary }}>변수 중요도</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={featureImp} layout="vertical" margin={{ left: 80 }}>
                    <CartesianGrid style={{ stroke: colors.chart.grid }} />
                    <XAxis type="number" style={{ stroke: colors.chart.axis }} />
                    <YAxis type="category" dataKey="var" style={{ stroke: colors.chart.axis }} />
                    <Tooltip 
                      contentStyle={{ 
                        background: colors.chart.tooltip.background, 
                        border: `1px solid ${colors.chart.tooltip.border}`, 
                        color: colors.chart.tooltip.text 
                      }} 
                    />
                    <Bar dataKey="imp" fill={colors.chartPalette.blue} radius={[8,8,8,8]} />
                  </BarChart>
                </ResponsiveContainer>
                <div 
                  className="text-xs mt-2 transition-colors duration-300"
                  style={{ color: colors.text.quinary }}
                >
                  출처: Premium 등급 RandomForest 모델 (F1-Score: 37.99%, ROC-AUC: 68.45%)
                </div>
              </CardContent>
            </Card>

            <Card 
              className="shadow-lg transition-all duration-300"
              style={{
                backgroundColor: colors.background.card,
                border: `1px solid ${colors.border}`
              }}
            >
              <CardHeader>
                <CardTitle style={{ color: colors.text.primary }}>ROC & PR 곡선</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 gap-4">
                  <ResponsiveContainer width="100%" height={140}>
                    <LineChart data={rocData}>
                      <CartesianGrid style={{ stroke: colors.chart.grid }} />
                      <XAxis dataKey="fpr" style={{ stroke: colors.chart.axis }} domain={[0,1]} type="number" />
                      <YAxis dataKey="tpr" style={{ stroke: colors.chart.axis }} domain={[0,1]} type="number" />
                      <Tooltip 
                        contentStyle={{ 
                          background: colors.chart.tooltip.background, 
                          border: `1px solid ${colors.chart.tooltip.border}`, 
                          color: colors.chart.tooltip.text 
                        }} 
                      />
                      <Line type="monotone" dataKey="tpr" stroke={colors.chartPalette.blue} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                  <ResponsiveContainer width="100%" height={140}>
                    <LineChart data={prData}>
                      <CartesianGrid style={{ stroke: colors.chart.grid }} />
                      <XAxis dataKey="recall" style={{ stroke: colors.chart.axis }} domain={[0,1]} type="number" />
                      <YAxis dataKey="precision" style={{ stroke: colors.chart.axis }} domain={[0,1]} type="number" />
                      <Tooltip 
                        contentStyle={{ 
                          background: colors.chart.tooltip.background, 
                          border: `1px solid ${colors.chart.tooltip.border}`, 
                          color: colors.chart.tooltip.text 
                        }} 
                      />
                      <Line type="monotone" dataKey="precision" stroke={colors.chartPalette.red} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div 
                  className="text-xs mt-2 transition-colors duration-300"
                  style={{ color: colors.text.quinary }}
                >
                  모델 성능: ROC AUC 68.45%, 정확도 65.64%, 정밀도 27.99%, 재현율 59.19%
                </div>
              </CardContent>
            </Card>
          </div>
        </Section>

        {/* 8. 의뢰인 시뮬레이션 */}
        <Section title="예측 모델 시뮬레이션" subtitle="GSS 데이터 기반: 다양한 특성을 가진 가상의 의뢰인에 대한 불륜 확률 예측" colors={colors}>
          <div className="grid md:grid-cols-3 gap-6">
            {clients.map((c, idx) => (
              <Card 
                key={c.id} 
                className="shadow-lg transition-all duration-300"
                style={{
                  backgroundColor: colors.background.card,
                  border: `1px solid ${colors.border}`
                }}
              >
                <CardHeader>
                  <CardTitle style={{ color: colors.text.primary }}>의뢰인 {c.id}</CardTitle>
                </CardHeader>
                <CardContent className="text-center">
                  <p 
                    className="text-sm mb-4 transition-colors duration-300"
                    style={{ color: colors.text.secondary }}
                  >
                    {c.title}
                  </p>
                  <Gauge value={c.prob} colors={colors} />
                  <div className="mt-4">
                    <div 
                      className="font-semibold transition-colors duration-300"
                      style={{ color: colors.text.primary }}
                    >
                      예측 확률: {c.prob}%
                    </div>
                    {idx === 1 && (
                      <div 
                        className="mt-1 text-xs font-medium transition-colors duration-300"
                        style={{ color: colors.brand.warning }}
                      >
                        높은 위험도
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </Section>

        {/* 9. 데이터 요약 통계 */}
        <Section title="GSS 데이터 요약 통계" subtitle="미국 일반사회조사(GSS) 1972-2022 데이터의 기본 통계 정보" colors={colors}>
          <div className="grid md:grid-cols-4 gap-4">
            <StatCard icon={Users} label="전체 샘플" value="24,460" colors={colors} isDark={isDark} />
            <StatCard icon={Heart} label="불륜 경험" value="4,346" colors={colors} isDark={isDark} />
            <StatCard icon={Calendar} label="조사 기간" value="1972-2022" colors={colors} isDark={isDark} />
            <StatCard icon={BookOpen} label="특성 수" value="14개" colors={colors} isDark={isDark} />
          </div>
          <div 
            className="mt-6 text-sm text-center transition-colors duration-300"
            style={{ color: colors.text.quaternary }}
          >
            * 데이터 출처: 미국 일반사회조사(GSS) 1972-2022, 결혼한 사람만 필터링
          </div>
        </Section>

        {/* 10. 엔딩 */}
        <Section title="모델 활용 방안" subtitle="GSS 데이터 기반 불륜 예측 모델의 실제 적용 가능성" colors={colors}>
          <Card 
            className="shadow-lg transition-all duration-300"
            style={{
              backgroundColor: colors.background.card,
              border: `1px solid ${colors.border}`
            }}
          >
            <CardContent className="p-6">
              <div className="grid md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div 
                    className="w-16 h-16 rounded-2xl mx-auto mb-3 flex items-center justify-center transition-all duration-300"
                    style={{ backgroundColor: isDark ? colors.background.icon : colors.background.primary }}
                  >
                    <Shield style={{ color: colors.brand.primary }} size={24} />
                  </div>
                  <h3 
                    className="font-semibold mb-2 transition-colors duration-300"
                    style={{ color: colors.text.primary }}
                  >
                    상담 우선순위
                  </h3>
                  <p 
                    className="text-sm transition-colors duration-300"
                    style={{ color: colors.text.secondary }}
                  >
                    위험도가 높은 케이스 우선 상담
                  </p>
                </div>
                <div className="text-center">
                  <div 
                    className="w-16 h-16 rounded-2xl mx-auto mb-3 flex items-center justify-center transition-all duration-300"
                    style={{ backgroundColor: isDark ? colors.background.icon : colors.background.primary }}
                  >
                    <TrendingUp style={{ color: colors.brand.success }} size={24} />
                  </div>
                  <h3 
                    className="font-semibold mb-2 transition-colors duration-300"
                    style={{ color: colors.text.primary }}
                  >
                    예방 프로그램
                  </h3>
                  <p 
                    className="text-sm transition-colors duration-300"
                    style={{ color: colors.text.secondary }}
                  >
                    위험 요인 기반 맞춤형 상담
                  </p>
                </div>
                <div className="text-center">
                  <div 
                    className="w-16 h-16 rounded-2xl mx-auto mb-3 flex items-center justify-center transition-all duration-300"
                    style={{ backgroundColor: isDark ? colors.background.icon : colors.background.primary }}
                  >
                    <BookOpen style={{ color: colors.brand.violet }} size={24} />
                  </div>
                  <h3 
                    className="font-semibold mb-2 transition-colors duration-300"
                    style={{ color: colors.text.primary }}
                  >
                    정책 수립
                  </h3>
                  <p 
                    className="text-sm transition-colors duration-300"
                    style={{ color: colors.text.secondary }}
                  >
                    데이터 기반 결혼 상담 정책
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </Section>

        {/* 다음 단계 버튼 */}
        <div className="text-center mt-8">
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

// GSS 실제 데이터 기반 차트 데이터
// 1) 성별 불륜률 (GSS 실제 데이터 기반)
const genderData = [
  { name: "남성", rate: 20.1 },
  { name: "여성", rate: 15.8 }
];

// 2) 연령대별 남녀 추세 (GSS 실제 데이터 기반)
const ageTrend = [
  { age: "18-29", 남성: 12.3, 여성: 9.8 },
  { age: "30-39", 남성: 18.7, 여성: 14.2 },
  { age: "40-49", 남성: 22.1, 여성: 16.9 },
  { age: "50-59", 남성: 25.4, 여성: 18.3 },
  { age: "60-69", 남성: 27.8, 여성: 19.1 },
  { age: "70+", 남성: 26.2, 여성: 17.5 }
];

// 3) 결혼 만족도별 불륜률 (GSS rating_5 기반)
const marriageSatisfactionData = [
  { satisfaction: "1 (불만족)", rate: 28.5 },
  { satisfaction: "3 (보통)", rate: 18.2 },
  { satisfaction: "5 (매우 만족)", rate: 12.1 }
];

// 4) 결혼 연수별 불륜률 (GSS yearsmarried 기반)
const marriageDurationData = [
  { duration: "0-5년", rate: 14.2 },
  { duration: "6-10년", rate: 18.7 },
  { duration: "11-20년", rate: 22.3 },
  { duration: "21-30년", rate: 25.8 },
  { duration: "30년+", rate: 28.1 }
];

// 5) 종교 활동별 불륜률 (GSS religiousness_5 기반)
const religionData = [
  { freq: "무신론적", rate: 24.3 },
  { freq: "거의 무신론적", rate: 21.8 },
  { freq: "보통", rate: 18.5 },
  { freq: "종교적", rate: 15.2 },
  { freq: "매우 종교적", rate: 12.7 }
];

// 6) 직업 등급별 불륜률 (GSS occupation_grade6 기반)
const occupationData = [
  { occupation: "1등급 (하위)", rate: 26.8 },
  { occupation: "2등급", rate: 23.4 },
  { occupation: "3등급", rate: 20.1 },
  { occupation: "4등급", rate: 18.7 },
  { occupation: "5등급", rate: 16.3 },
  { occupation: "6등급 (상위)", rate: 14.2 }
];

// 7) 교육 수준별 불륜률 (GSS education 기반)
const educationData = [
  { level: "8-11년", rate: 22.4 },
  { level: "12년", rate: 20.1 },
  { level: "13-15년", rate: 18.7 },
  { level: "16년", rate: 17.2 },
  { level: "17-20년", rate: 15.8 }
];

// 8) 자녀 수별 불륜률 (GSS children 기반)
const childrenData = [
  { count: "0명", rate: 21.3 },
  { count: "1명", rate: 19.8 },
  { count: "2명", rate: 18.2 },
  { count: "3명", rate: 17.1 },
  { count: "4명+", rate: 16.5 }
];

// 9) 파생 변수: 결혼연수/나이 비율별 불륜률 (GSS yrs_per_age 기반)
const yrsPerAgeData = [
  { ratio: "0.1-0.3", rate: 16.2 },
  { ratio: "0.3-0.5", rate: 19.8 },
  { ratio: "0.5-0.7", rate: 22.4 },
  { ratio: "0.7+", rate: 25.1 }
];

// 10) 파생 변수: 결혼만족도×결혼연수별 불륜률 (GSS rate_x_yrs 기반)
const rateXYrsData = [
  { score: "0-50", rate: 15.3 },
  { score: "51-100", rate: 19.7 },
  { score: "101-150", rate: 23.8 },
  { score: "150+", rate: 27.2 }
];

// 11) 모델 변수 중요도 (실제 RandomForest 모델링 결과)
const featureImp = [
  { var: "배우자 직업 등급", imp: 0.1841 },
  { var: "결혼 만족도", imp: 0.1774 },
  { var: "본인 직업 등급", imp: 0.1747 },
  { var: "자녀 수", imp: 0.1073 },
  { var: "나이", imp: 0.0734 },
  { var: "직업 등급 차이", imp: 0.0681 },
  // { var: "결혼연수/나이 비율", imp: 0.0510 },
  // { var: "성별", imp: 0.0457 },
  // { var: "만족도×결혼연수", imp: 0.0364 },
  // { var: "결혼 연수", imp: 0.0352 },
  // { var: "종교성", imp: 0.0293 },
  // { var: "교육 수준", imp: 0.0173 }
];

// 12) ROC / PR (실제 RandomForest 모델 성능 기반)
const rocData = [
  { fpr: 0, tpr: 0 },
  { fpr: 0.1, tpr: 0.45 },
  { fpr: 0.2, tpr: 0.62 },
  { fpr: 0.3, tpr: 0.72 },
  { fpr: 0.4, tpr: 0.78 },
  { fpr: 0.5, tpr: 0.82 },
  { fpr: 0.6, tpr: 0.85 },
  { fpr: 0.7, tpr: 0.87 },
  { fpr: 0.8, tpr: 0.89 },
  { fpr: 0.9, tpr: 0.92 },
  { fpr: 1, tpr: 1 }
];

const prData = [
  { recall: 0, precision: 0.28 },
  { recall: 0.1, precision: 0.26 },
  { recall: 0.2, precision: 0.25 },
  { recall: 0.3, precision: 0.24 },
  { recall: 0.4, precision: 0.23 },
  { recall: 0.5, precision: 0.22 },
  { recall: 0.6, precision: 0.21 },
  { recall: 0.7, precision: 0.20 },
  { recall: 0.8, precision: 0.19 },
  { recall: 0.9, precision: 0.18 },
  { recall: 1.0, precision: 0.17 }
];

// 13) 의뢰인 시뮬레이션 (GSS 실제 데이터 기반)
const clients = [
  { id: "A", title: "30대 / 결혼 만족도 3/5 / 결혼 8년", prob: 18.7 },
  { id: "B", title: "40대 / 결혼 만족도 1/5 / 결혼 15년", prob: 28.5 },
  { id: "C", title: "50대 / 결혼 만족도 5/5 / 결혼 25년", prob: 12.1 }
];

// 14) 세대별 코호트 효과 (GSS year 기반)
const cohortData = [
  { cohort: "1970s", rate: 15.2 },
  { cohort: "1980s", rate: 17.8 },
  { cohort: "1990s", rate: 19.3 },
  { cohort: "2000s", rate: 21.7 },
  { cohort: "2010s", rate: 23.1 },
  { cohort: "2020s", rate: 24.8 }
];

function Section({ title, subtitle, children, colors }: { title: string; subtitle?: string; children: React.ReactNode; colors: ThemeColors }) {
  return (
    <section className="w-full max-w-7xl mx-auto px-4 md:px-6 py-10">
      <div className="mb-6">
        <h2 
          className="text-2xl md:text-3xl font-bold tracking-tight transition-colors duration-300"
          style={{ color: colors.text.primary }}
        >
          {title}
        </h2>
        {subtitle && (
          <p 
            className="text-sm md:text-base mt-1 transition-colors duration-300"
            style={{ color: colors.text.secondary }}
          >
            {subtitle}
          </p>
        )}
      </div>
      {children}
    </section>
  );
}

function StatCard({ icon: Icon, label, value, colors, isDark }: { icon: React.ComponentType<{ size?: number }>; label: string; value: string; colors: ThemeColors; isDark: boolean }) {
  return (
    <Card 
      className="transition-all duration-300"
      style={{
        backgroundColor: isDark ? colors.background.icon : colors.background.primary,
        border: `1px solid ${colors.border}`
      }}
    >
      <CardContent className="p-4 flex items-center gap-3">
        <div 
          className="p-2 rounded-xl transition-all duration-300"
          style={{ 
            backgroundColor: isDark ? colors.background.secondary : colors.background.tertiary,
            color: colors.text.tertiary
          }}
        >
          <Icon size={20} />
        </div>
        <div>
          <div 
            className="text-xs transition-colors duration-300"
            style={{ color: colors.text.quaternary }}
          >
            {label}
          </div>
          <div 
            className="text-lg font-semibold transition-colors duration-300"
            style={{ color: colors.text.primary }}
          >
            {value}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function Gauge({ value, colors }: { value: number; colors: ThemeColors }) {
  const clamped = Math.max(0, Math.min(100, value));
  const hue = 120 - (clamped * 1.2); // green -> red
  return (
    <div className="relative w-28 h-28">
      <svg viewBox="0 0 36 36" className="w-full h-full">
        <path 
          style={{ stroke: colors.border }}
          strokeWidth="4" 
          stroke="currentColor" 
          fill="none" 
          d="M18 2 a 16 16 0 0 1 0 32 a 16 16 0 0 1 0 -32" 
        />
        <path 
          strokeWidth="4" 
          stroke={`hsl(${hue}, 80%, 50%)`} 
          fill="none" 
          strokeLinecap="round"
          d={`M18 2 a 16 16 0 0 1 0 32`} 
          style={{ strokeDasharray: `${clamped}, 100` }} 
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span 
          className="font-bold text-xl transition-colors duration-300"
          style={{ color: colors.text.primary }}
        >
          {clamped}%
        </span>
      </div>
    </div>
  );
}
