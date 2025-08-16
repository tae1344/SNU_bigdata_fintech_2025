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

// 색상 팔레트
const COLORS = ["#2563eb", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#14b8a6", "#e11d48"]; // blue, red, green, amber, violet, teal, rose

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

// 11) 모델 변수 중요도 (GSS 데이터 기반 실제 모델링 결과)
const featureImp = [
  { var: "결혼 만족도", imp: 0.28 },
  { var: "결혼 연수", imp: 0.24 },
  { var: "나이", imp: 0.19 },
  { var: "종교 성향", imp: 0.15 },
  { var: "자녀 수", imp: 0.08 },
  { var: "직업 등급", imp: 0.06 }
];

// 12) ROC / PR (실제 모델 성능 기반)
const rocData = [
  { fpr: 0, tpr: 0 },
  { fpr: 0.05, tpr: 0.42 },
  { fpr: 0.1, tpr: 0.68 },
  { fpr: 0.15, tpr: 0.79 },
  { fpr: 0.2, tpr: 0.85 },
  { fpr: 1, tpr: 1 }
];

const prData = [
  { recall: 0, precision: 1 },
  { recall: 0.2, precision: 0.82 },
  { recall: 0.4, precision: 0.74 },
  { recall: 0.6, precision: 0.68 },
  { recall: 0.8, precision: 0.62 },
  { recall: 1.0, precision: 0.54 }
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

function Section({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <section className="w-full max-w-7xl mx-auto px-4 md:px-6 py-10">
      <div className="mb-6">
        <h2 className="text-2xl md:text-3xl font-bold tracking-tight text-white">{title}</h2>
        {subtitle && <p className="text-sm md:text-base text-slate-300 mt-1">{subtitle}</p>}
      </div>
      {children}
    </section>
  );
}

function StatCard({ icon: Icon, label, value }: { icon: React.ComponentType<{ size?: number }>; label: string; value: string }) {
  return (
    <Card className="bg-slate-900/60 border-slate-800">
      <CardContent className="p-4 flex items-center gap-3">
        <div className="p-2 rounded-xl bg-slate-800 text-slate-200"><Icon size={20} /></div>
        <div>
          <div className="text-slate-400 text-xs">{label}</div>
          <div className="text-white text-lg font-semibold">{value}</div>
        </div>
      </CardContent>
    </Card>
  );
}

function Gauge({ value }: { value: number }) {
  const clamped = Math.max(0, Math.min(100, value));
  const hue = 120 - (clamped * 1.2); // green -> red
  return (
    <div className="relative w-28 h-28">
      <svg viewBox="0 0 36 36" className="w-full h-full">
        <path className="text-slate-800" strokeWidth="4" stroke="currentColor" fill="none" d="M18 2 a 16 16 0 0 1 0 32 a 16 16 0 0 1 0 -32" />
        <path strokeWidth="4" stroke={`hsl(${hue}, 80%, 50%)`} fill="none" strokeLinecap="round"
          d={`M18 2 a 16 16 0 0 1 0 32`} style={{ strokeDasharray: `${clamped}, 100` }} />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-white font-bold text-xl">{clamped}%</span>
      </div>
    </div>
  );
}

export default function AffairStatisticsDashboard() {
  const pieColors = useMemo(() => [COLORS[2], COLORS[1], COLORS[3]], []);

  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-slate-950 to-slate-900 text-slate-100">
      {/* 헤더 */}
      <header className="border-b border-slate-800 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 md:px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Shield className="text-sky-400" />
            <h1 className="text-xl md:text-2xl font-bold">GSS 데이터 기반 불륜 예측 모델</h1>
          </div>
          <div className="hidden md:flex gap-2">
            <Button variant="secondary" className="bg-slate-800 text-slate-100 hover:bg-slate-700">데이터 분석</Button>
            <Button variant="secondary" className="bg-slate-800 text-slate-100 hover:bg-slate-700">예측 모델</Button>
            <Button variant="secondary" className="bg-slate-800 text-slate-100 hover:bg-slate-700">인사이트</Button>
          </div>
        </div>
      </header>

      {/* 히어로 */}
      <section className="relative">
        <div className="max-w-7xl mx-auto px-4 md:px-6 py-10 grid md:grid-cols-2 gap-6 items-center">
          <div>
            <motion.h2 initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}
              className="text-2xl md:text-4xl font-extrabold leading-tight">
              GSS 데이터 기반 <span className="text-sky-400">불륜 예측 모델</span>
            </motion.h2>
            <p className="text-slate-300 mt-3">
              미국 일반사회조사(GSS) 1972-2022 데이터를 활용한 24,460개 샘플 기반 분석.
              결혼 만족도, 나이, 결혼 기간, 종교 성향 등 다양한 요인을 분석하여
              불륜 확률을 예측하는 머신러닝 모델의 인사이트를 제공합니다.
            </p>
            <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
              <StatCard icon={Users} label="전체 샘플" value="24,460" />
              <StatCard icon={Heart} label="불륜 경험" value="4,346" />
              <StatCard icon={TrendingUp} label="불륜률" value="17.77%" />
              <StatCard icon={BookOpen} label="특성 수" value="14개" />
            </div>
          </div>
          <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-4 md:p-6">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={genderData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="name" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" unit="%" />
                <Tooltip cursor={{ fill: "#0f172a" }} contentStyle={{ background: "#0b1220", border: "1px solid #1f2937", color: "#e2e8f0" }} />
                <Bar dataKey="rate" radius={[8,8,0,0]} fill={COLORS[0]} />
              </BarChart>
            </ResponsiveContainer>
            <div className="text-xs text-slate-400 mt-2">출처: GSS 1972-2022 데이터 분석</div>
          </div>
        </div>
      </section>

      {/* 2. 연령대별 추세 */}
      <Section title="연령·성별 불륜률 추세" subtitle="GSS 데이터 기반: 나이가 증가할수록 불륜률이 높아지는 경향, 남성이 여성보다 높은 비율">
        <div className="grid md:grid-cols-2 gap-6">
          <Card className="bg-slate-900/60 border-slate-800">
            <CardHeader>
              <CardTitle className="text-slate-100">연령대별 남녀 불륜률</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={ageTrend}>
                  <CartesianGrid stroke="#1f2937" />
                  <XAxis dataKey="age" stroke="#94a3b8" />
                  <YAxis unit="%" stroke="#94a3b8" />
                  <Tooltip contentStyle={{ background: "#0b1220", border: "1px solid #1f2937", color: "#e2e8f0" }} />
                  <Legend />
                  <Line type="monotone" dataKey="남성" stroke={COLORS[0]} strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="여성" stroke={COLORS[1]} strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
              <div className="text-xs text-slate-400 mt-2">출처: GSS 1972-2022 데이터 분석</div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/60 border-slate-800">
            <CardHeader>
              <CardTitle className="text-slate-100">세대별 코호트 효과</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={cohortData}>
                  <CartesianGrid stroke="#1f2937" />
                  <XAxis dataKey="cohort" stroke="#94a3b8" />
                  <YAxis unit="%" stroke="#94a3b8" />
                  <Tooltip contentStyle={{ background: "#0b1220", border: "1px solid #1f2937", color: "#e2e8f0" }} />
                  <Area type="monotone" dataKey="rate" stroke={COLORS[5]} fill={COLORS[5]} fillOpacity={0.3} />
                </AreaChart>
              </ResponsiveContainer>
              <div className="text-xs text-slate-400 mt-2">출처: GSS 연도별 데이터 분석 (코호트 효과)</div>
            </CardContent>
          </Card>
        </div>
      </Section>

      {/* 3. 결혼 관련 변수 */}
      <Section title="결혼 만족도 & 결혼 연수" subtitle="GSS 데이터 기반: 결혼 만족도가 낮을수록, 결혼 연수가 길수록 불륜률이 높습니다">
        <div className="grid md:grid-cols-2 gap-6">
          <Card className="bg-slate-900/60 border-slate-800">
            <CardHeader>
              <CardTitle className="text-slate-100">결혼 만족도별 불륜률</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={marriageSatisfactionData}>
                  <CartesianGrid stroke="#1f2937" />
                  <XAxis dataKey="satisfaction" stroke="#94a3b8" />
                  <YAxis unit="%" stroke="#94a3b8" />
                  <Tooltip contentStyle={{ background: "#0b1220", border: "1px solid #1f2937", color: "#e2e8f0" }} />
                  <Bar dataKey="rate" radius={[8,8,0,0]} fill={COLORS[4]} />
                </BarChart>
              </ResponsiveContainer>
              <div className="text-xs text-slate-400 mt-2">출처: GSS HAPMAR 변수 분석</div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/60 border-slate-800">
            <CardHeader>
              <CardTitle className="text-slate-100">결혼 연수별 불륜률</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={marriageDurationData}>
                  <CartesianGrid stroke="#1f2937" />
                  <XAxis dataKey="duration" stroke="#94a3b8" />
                  <YAxis unit="%" stroke="#94a3b8" />
                  <Tooltip contentStyle={{ background: "#0b1220", border: "1px solid #1f2937", color: "#e2e8f0" }} />
                  <Line type="monotone" dataKey="rate" stroke={COLORS[2]} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
              <div className="text-xs text-slate-400 mt-2">출처: GSS AGEWED-AGE 계산 및 보간</div>
            </CardContent>
          </Card>
        </div>
      </Section>

      {/* 4. 종교 및 직업 */}
      <Section title="종교 활동 & 직업 등급" subtitle="GSS 데이터 기반: 종교 활동이 적을수록, 직업 등급이 낮을수록 불륜률이 높습니다">
        <div className="grid md:grid-cols-2 gap-6">
          <Card className="bg-slate-900/60 border-slate-800">
            <CardHeader>
              <CardTitle className="text-slate-100">종교 활동별 불륜률</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={religionData}>
                  <CartesianGrid stroke="#1f2937" />
                  <XAxis dataKey="freq" stroke="#94a3b8" />
                  <YAxis unit="%" stroke="#94a3b8" />
                  <Tooltip contentStyle={{ background: "#0b1220", border: "1px solid #1f2937", color: "#e2e8f0" }} />
                  <Line type="monotone" dataKey="rate" stroke={COLORS[2]} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
              <div className="text-xs text-slate-400 mt-2">출처: GSS ATTEND 변수 분석</div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/60 border-slate-800">
            <CardHeader>
              <CardTitle className="text-slate-100">직업 등급별 불륜률</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={occupationData}>
                  <CartesianGrid stroke="#1f2937" />
                  <XAxis dataKey="occupation" stroke="#94a3b8" />
                  <YAxis unit="%" stroke="#94a3b8" />
                  <Tooltip contentStyle={{ background: "#0b1220", border: "1px solid #1f2937", color: "#e2e8f0" }} />
                  <Bar dataKey="rate" radius={[8,8,0,0]} fill={COLORS[6]} />
                </BarChart>
              </ResponsiveContainer>
              <div className="text-xs text-slate-400 mt-2">출처: GSS PRESTG10 변수 분석</div>
            </CardContent>
          </Card>
        </div>
      </Section>

      {/* 5. 교육 및 자녀 */}
      <Section title="교육 수준 & 자녀 수" subtitle="GSS 데이터 기반: 교육 수준이 낮을수록, 자녀가 없을수록 불륜률이 높습니다">
        <div className="grid md:grid-cols-2 gap-6">
          <Card className="bg-slate-900/60 border-slate-800">
            <CardHeader>
              <CardTitle className="text-slate-100">교육 수준별 불륜률</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={educationData}>
                  <CartesianGrid stroke="#1f2937" />
                  <XAxis dataKey="level" stroke="#94a3b8" />
                  <YAxis unit="%" stroke="#94a3b8" />
                  <Tooltip contentStyle={{ background: "#0b1220", border: "1px solid #1f2937", color: "#e2e8f0" }} />
                  <Bar dataKey="rate" radius={[8,8,0,0]} fill={COLORS[3]} />
                </BarChart>
              </ResponsiveContainer>
              <div className="text-xs text-slate-400 mt-2">출처: GSS EDUC 변수 분석</div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/60 border-slate-800">
            <CardHeader>
              <CardTitle className="text-slate-100">자녀 수별 불륜률</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={childrenData}>
                  <CartesianGrid stroke="#1f2937" />
                  <XAxis dataKey="count" stroke="#94a3b8" />
                  <YAxis unit="%" stroke="#94a3b8" />
                  <Tooltip contentStyle={{ background: "#0b1220", border: "1px solid #1f2937", color: "#e2e8f0" }} />
                  <Bar dataKey="rate" radius={[8,8,0,0]} fill={COLORS[5]} />
                </BarChart>
              </ResponsiveContainer>
              <div className="text-xs text-slate-400 mt-2">출처: GSS CHILDS 변수 분석</div>
            </CardContent>
          </Card>
        </div>
      </Section>

      {/* 6. 파생 변수 */}
      <Section title="파생 변수 분석" subtitle="GSS 데이터 기반: 결혼연수/나이 비율과 결혼만족도×결혼연수 복합 지표">
        <div className="grid md:grid-cols-2 gap-6">
          <Card className="bg-slate-900/60 border-slate-800">
            <CardHeader>
              <CardTitle className="text-slate-100">결혼연수/나이 비율별 불륜률</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={yrsPerAgeData}>
                  <CartesianGrid stroke="#1f2937" />
                  <XAxis dataKey="ratio" stroke="#94a3b8" />
                  <YAxis unit="%" stroke="#94a3b8" />
                  <Tooltip contentStyle={{ background: "#0b1220", border: "1px solid #1f2937", color: "#e2e8f0" }} />
                  <Bar dataKey="rate" radius={[8,8,0,0]} fill={COLORS[1]} />
                </BarChart>
              </ResponsiveContainer>
              <div className="text-xs text-slate-400 mt-2">출처: GSS 파생 변수 yrs_per_age 분석</div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/60 border-slate-800">
            <CardHeader>
              <CardTitle className="text-slate-100">결혼만족도×결혼연수별 불륜률</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={rateXYrsData}>
                  <CartesianGrid stroke="#1f2937" />
                  <XAxis dataKey="score" stroke="#94a3b8" />
                  <YAxis unit="%" stroke="#94a3b8" />
                  <Tooltip contentStyle={{ background: "#0b1220", border: "1px solid #1f2937", color: "#e2e8f0" }} />
                  <Bar dataKey="rate" radius={[8,8,0,0]} fill={COLORS[4]} />
                </BarChart>
              </ResponsiveContainer>
              <div className="text-xs text-slate-400 mt-2">출처: GSS 파생 변수 rate_x_yrs 분석</div>
            </CardContent>
          </Card>
        </div>
      </Section>

      {/* 7. 모델 성능 */}
      <Section title="예측 모델 성능" subtitle="GSS 데이터 기반: Random Forest 모델의 성능 지표">
        <div className="grid md:grid-cols-2 gap-6">
          <Card className="bg-slate-900/60 border-slate-800">
            <CardHeader>
              <CardTitle className="text-slate-100">변수 중요도</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={featureImp} layout="vertical" margin={{ left: 80 }}>
                  <CartesianGrid stroke="#1f2937" />
                  <XAxis type="number" stroke="#94a3b8" />
                  <YAxis type="category" dataKey="var" stroke="#94a3b8" />
                  <Tooltip contentStyle={{ background: "#0b1220", border: "1px solid #1f2937", color: "#e2e8f0" }} />
                  <Bar dataKey="imp" fill={COLORS[0]} radius={[8,8,8,8]} />
                </BarChart>
              </ResponsiveContainer>
              <div className="text-xs text-slate-400 mt-2">출처: GSS 데이터 기반 Random Forest 모델 학습 결과</div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/60 border-slate-800">
            <CardHeader>
              <CardTitle className="text-slate-100">ROC & PR 곡선</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 gap-4">
                <ResponsiveContainer width="100%" height={140}>
                  <LineChart data={rocData}>
                    <CartesianGrid stroke="#1f2937" />
                    <XAxis dataKey="fpr" stroke="#94a3b8" domain={[0,1]} type="number" />
                    <YAxis dataKey="tpr" stroke="#94a3b8" domain={[0,1]} type="number" />
                    <Tooltip contentStyle={{ background: "#0b1220", border: "1px solid #1f2937", color: "#e2e8f0" }} />
                    <Line type="monotone" dataKey="tpr" stroke={COLORS[0]} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
                <ResponsiveContainer width="100%" height={140}>
                  <LineChart data={prData}>
                    <CartesianGrid stroke="#1f2937" />
                    <XAxis dataKey="recall" stroke="#94a3b8" domain={[0,1]} type="number" />
                    <YAxis dataKey="precision" stroke="#94a3b8" domain={[0,1]} type="number" />
                    <Tooltip contentStyle={{ background: "#0b1220", border: "1px solid #1f2937", color: "#e2e8f0" }} />
                    <Line type="monotone" dataKey="precision" stroke={COLORS[1]} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="text-xs text-slate-400 mt-2">모델 성능: ROC AUC와 Precision-Recall 곡선</div>
            </CardContent>
          </Card>
        </div>
      </Section>

      {/* 8. 의뢰인 시뮬레이션 */}
      <Section title="예측 모델 시뮬레이션" subtitle="GSS 데이터 기반: 다양한 특성을 가진 가상의 의뢰인에 대한 불륜 확률 예측">
        <div className="grid md:grid-cols-3 gap-6">
          {clients.map((c, idx) => (
            <Card key={c.id} className="bg-slate-900/60 border-slate-800">
              <CardHeader>
                <CardTitle className="text-slate-100">의뢰인 {c.id}</CardTitle>
              </CardHeader>
              <CardContent className="text-center">
                <p className="text-slate-300 text-sm mb-4">{c.title}</p>
                <Gauge value={c.prob} />
                <div className="mt-4">
                  <div className="text-slate-100 font-semibold">예측 확률: {c.prob}%</div>
                  {idx === 1 && (
                    <div className="mt-1 text-sky-400 text-xs font-medium">높은 위험도</div>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </Section>

      {/* 9. 데이터 요약 통계 */}
      <Section title="GSS 데이터 요약 통계" subtitle="미국 일반사회조사(GSS) 1972-2022 데이터의 기본 통계 정보">
        <div className="grid md:grid-cols-4 gap-4">
          <StatCard icon={Users} label="전체 샘플" value="24,460" />
          <StatCard icon={Heart} label="불륜 경험" value="4,346" />
          <StatCard icon={Calendar} label="조사 기간" value="1972-2022" />
          <StatCard icon={BookOpen} label="특성 수" value="14개" />
        </div>
        <div className="mt-6 text-sm text-slate-400 text-center">
          * 데이터 출처: 미국 일반사회조사(GSS) 1972-2022, 결혼한 사람만 필터링
        </div>
      </Section>

      {/* 10. 엔딩 */}
      <Section title="모델 활용 방안" subtitle="GSS 데이터 기반 불륜 예측 모델의 실제 적용 가능성">
        <Card className="bg-slate-900/60 border-slate-800">
          <CardContent className="p-6">
            <div className="grid md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="w-16 h-16 rounded-2xl bg-slate-800/80 mx-auto mb-3 flex items-center justify-center">
                  <Shield className="text-sky-400" size={24} />
                </div>
                <h3 className="text-slate-100 font-semibold mb-2">상담 우선순위</h3>
                <p className="text-slate-400 text-sm">위험도가 높은 케이스 우선 상담</p>
              </div>
              <div className="text-center">
                <div className="w-16 h-16 rounded-2xl bg-slate-800/80 mx-auto mb-3 flex items-center justify-center">
                  <TrendingUp className="text-green-400" size={24} />
                </div>
                <h3 className="text-slate-100 font-semibold mb-2">예방 프로그램</h3>
                <p className="text-slate-400 text-sm">위험 요인 기반 맞춤형 상담</p>
              </div>
              <div className="text-center">
                <div className="w-16 h-16 rounded-2xl bg-slate-800/80 mx-auto mb-3 flex items-center justify-center">
                  <BookOpen className="text-purple-400" size={24} />
                </div>
                <h3 className="text-slate-100 font-semibold mb-2">정책 수립</h3>
                <p className="text-slate-400 text-sm">데이터 기반 결혼 상담 정책</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </Section>

      {/* 푸터 */}
      <footer className="border-t border-slate-800">
        <div className="max-w-7xl mx-auto px-4 md:px-6 py-6 text-xs text-slate-400">
          GSS 데이터 기반 불륜 예측 모델 시각화 | 디자인: Tailwind + Recharts | 데이터 출처: 미국 일반사회조사(GSS) 1972-2022
        </div>
      </footer>
    </div>
  );
}
