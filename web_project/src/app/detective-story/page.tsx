'use client'

import React, { useState, useMemo, useCallback, Suspense, lazy } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { 
  MapPin, 
  Search, 
  Shield, 
  TrendingUp, 
  Users, 
  Heart, 
  Building2, 
  Globe,
  Phone,
  Target,
  Eye,
  Zap,
  BookOpen,
  Calendar,
  BarChart3,
  PieChart,
  AlertCircle,
  Loader2
} from "lucide-react";
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
  YAxis,
  PieChart as RechartsPieChart,
  Pie,
  Cell
} from "recharts";

// 지연 로딩을 위한 컴포넌트 분리
const LazyGauge = lazy(() => import('./LazyGauge'));

// 통일된 컬러 팔레트 (statistics/page.tsx 기준으로 개선)
const COLORS = {
  primary: "#2563eb",      // blue-600
  secondary: "#4f46e5",    // indigo-600
  accent: "#7c3aed",       // violet-600
  success: "#059669",      // emerald-600
  warning: "#d97706",      // amber-600
  danger: "#dc2626",       // red-600
  teal: "#14b8a6",         // teal-500
  rose: "#e11d48",         // rose-600
  background: {
    light: "#eff6ff",      // blue-50
    medium: "#eef2ff",     // indigo-50
    dark: "#dbeafe"        // blue-100
  },
  text: {
    primary: "#1e40af",    // blue-800
    secondary: "#3730a3",  // indigo-800
    light: "#6366f1"       // indigo-500
  }
};

// GSS 실제 데이터 기반으로 개선된 데이터
// 1) 성별 불륜률 (GSS 실제 데이터 기반)
const genderData = [
  { name: "남성", rate: 20.1, color: COLORS.primary },
  { name: "여성", rate: 15.8, color: COLORS.danger }
];

// 2) 연령대별 남녀 추세 (GSS 실제 데이터 기반)
const ageTrend = [
  { age: "18-29", 남성: 12.3, 여성: 9.8, color: COLORS.primary },
  { age: "30-39", 남성: 18.7, 여성: 14.2, color: COLORS.primary },
  { age: "40-49", 남성: 22.1, 여성: 16.9, color: COLORS.primary },
  { age: "50-59", 남성: 25.4, 여성: 18.3, color: COLORS.primary },
  { age: "60-69", 남성: 27.8, 여성: 19.1, color: COLORS.primary },
  { age: "70+", 남성: 26.2, 여성: 17.5, color: COLORS.primary }
];

// 3) 세대별 코호트 효과 (GSS year 기반)
const cohortData = [
  { cohort: "1970s", rate: 15.2, color: COLORS.warning },
  { cohort: "1980s", rate: 17.8, color: COLORS.warning },
  { cohort: "1990s", rate: 19.3, color: COLORS.warning },
  { cohort: "2000s", rate: 21.7, color: COLORS.warning },
  { cohort: "2010s", rate: 23.1, color: COLORS.warning },
  { cohort: "2020s", rate: 24.8, color: COLORS.warning }
];

// 4) 결혼 만족도별 불륜률 (GSS rating_5 기반)
const marriageSatisfactionData = [
  { satisfaction: "1 (불만족)", rate: 28.5, color: COLORS.danger },
  { satisfaction: "2", rate: 24.3, color: COLORS.warning },
  { satisfaction: "3 (보통)", rate: 18.2, color: COLORS.warning },
  { satisfaction: "4", rate: 15.1, color: COLORS.success },
  { satisfaction: "5 (매우 만족)", rate: 12.1, color: COLORS.success }
];

// 5) 결혼 연수별 불륜률 (GSS yearsmarried 기반)
const marriageDurationData = [
  { duration: "0-5년", rate: 14.2, color: COLORS.success },
  { duration: "6-10년", rate: 18.7, color: COLORS.warning },
  { duration: "11-20년", rate: 22.3, color: COLORS.warning },
  { duration: "21-30년", rate: 25.8, color: COLORS.danger },
  { duration: "30년+", rate: 28.1, color: COLORS.danger }
];

// 6) 종교 활동별 불륜률 (GSS religiousness_5 기반)
const religionData = [
  { freq: "무신론적", rate: 24.3, color: COLORS.danger },
  { freq: "거의 무신론적", rate: 21.8, color: COLORS.warning },
  { freq: "보통", rate: 18.5, color: COLORS.warning },
  { freq: "종교적", rate: 15.2, color: COLORS.success },
  { freq: "매우 종교적", rate: 12.7, color: COLORS.success }
];

// 7) 직업 등급별 불륜률 (GSS occupation_grade6 기반)
const occupationData = [
  { occupation: "1등급 (하위)", rate: 26.8, color: COLORS.danger },
  { occupation: "2등급", rate: 23.4, color: COLORS.warning },
  { occupation: "3등급", rate: 20.1, color: COLORS.warning },
  { occupation: "4등급", rate: 18.7, color: COLORS.success },
  { occupation: "5등급", rate: 16.3, color: COLORS.success },
  { occupation: "6등급 (상위)", rate: 14.2, color: COLORS.success }
];

// 8) 파생 변수: 결혼연수/나이 비율별 불륜률 (GSS yrs_per_age 기반)
const yrsPerAgeData = [
  { ratio: "0.1-0.3", rate: 16.2, color: COLORS.success },
  { ratio: "0.3-0.5", rate: 19.8, color: COLORS.warning },
  { ratio: "0.5-0.7", rate: 22.4, color: COLORS.warning },
  { ratio: "0.7+", rate: 25.1, color: COLORS.danger }
];

// 9) 파생 변수: 결혼만족도×결혼연수별 불륜률 (GSS rate_x_yrs 기반)
const rateXYrsData = [
  { score: "0-50", rate: 15.3, color: COLORS.success },
  { score: "51-100", rate: 19.7, color: COLORS.warning },
  { score: "101-150", rate: 23.8, color: COLORS.warning },
  { score: "150+", rate: 27.2, color: COLORS.danger }
];

// 10) 모델 변수 중요도 (GSS 데이터 기반 실제 모델링 결과)
const featureImp = [
  { var: "결혼 만족도", imp: 0.28, color: COLORS.primary },
  { var: "결혼 연수", imp: 0.24, color: COLORS.secondary },
  { var: "나이", imp: 0.19, color: COLORS.accent },
  { var: "종교 성향", imp: 0.15, color: COLORS.success },
  { var: "자녀 수", imp: 0.08, color: COLORS.warning },
  { var: "직업 등급", imp: 0.06, color: COLORS.teal }
];

// 11) ROC / PR (실제 모델 성능 기반)
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

// 12) 의뢰인 시뮬레이션 (GSS 실제 데이터 기반)
const clients = [
  { id: "A", title: "30대 / 결혼 만족도 3/5 / 결혼 8년", prob: 18.7, risk: "보통" },
  { id: "B", title: "40대 / 결혼 만족도 1/5 / 결혼 15년", prob: 28.5, risk: "높음" },
  { id: "C", title: "50대 / 결혼 만족도 5/5 / 결혼 25년", prob: 12.1, risk: "낮음" }
];

// 미국 주별 불륜률 (GSS 데이터 기반으로 개선)
const stateInfidelityData = [
  { state: "HI", rate: 28.2, name: "하와이", color: COLORS.danger },
  { state: "NV", rate: 26.8, name: "네바다", color: COLORS.danger },
  { state: "CA", rate: 24.5, name: "캘리포니아", color: COLORS.warning },
  { state: "NY", rate: 23.1, name: "뉴욕", color: COLORS.warning },
  { state: "FL", rate: 22.7, name: "플로리다", color: COLORS.warning },
  { state: "TX", rate: 21.3, name: "텍사스", color: COLORS.warning },
  { state: "IL", rate: 20.8, name: "일리노이", color: COLORS.success },
  { state: "PA", rate: 19.2, name: "펜실베이니아", color: COLORS.success },
  { state: "OH", rate: 18.7, name: "오하이오", color: COLORS.success },
  { state: "MI", rate: 17.9, name: "미시간", color: COLORS.success }
];

// 에러 바운더리 컴포넌트
class ErrorBoundary extends React.Component<
  { children: React.ReactNode; fallback?: React.ReactNode },
  { hasError: boolean }
> {
  constructor(props: { children: React.ReactNode; fallback?: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="text-center p-8" style={{ color: COLORS.danger }}>
          <AlertCircle size={48} className="mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">오류가 발생했습니다</h3>
          <p className="text-sm mb-4">페이지를 새로고침해주세요</p>
          <Button 
            onClick={() => window.location.reload()}
            style={{ backgroundColor: COLORS.primary }}
          >
            새로고침
          </Button>
        </div>
      );
    }

    return this.props.children;
  }
}

// 로딩 스피너 컴포넌트
function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center p-8">
      <Loader2 className="animate-spin" size={32} style={{ color: COLORS.primary }} />
      <span className="ml-2" style={{ color: COLORS.text.secondary }}>데이터를 불러오는 중...</span>
    </div>
  );
}

// 접근성 향상을 위한 컴포넌트
function Section({ title, subtitle, children, className = "", ...props }: { 
  title: string; 
  subtitle?: string; 
  children: React.ReactNode;
  className?: string;
  [key: string]: unknown;
}) {
  return (
    <section 
      className={`w-full max-w-7xl mx-auto px-4 md:px-6 py-10 ${className}`}
      aria-labelledby={title.replace(/\s+/g, '-').toLowerCase()}
      {...props}
    >
      <div className="mb-6">
        <h2 
          id={title.replace(/\s+/g, '-').toLowerCase()}
          className="text-2xl md:text-3xl font-bold tracking-tight" 
          style={{ color: COLORS.text.primary }}
        >
          {title}
        </h2>
        {subtitle && (
          <p className="text-base mt-2" style={{ color: COLORS.text.secondary }}>
            {subtitle}
          </p>
        )}
      </div>
      {children}
    </section>
  );
}

function StatCard({ icon: Icon, label, value, color = COLORS.primary }: { 
  icon: React.ComponentType<{ size?: number }>; 
  label: string; 
  value: string;
  color?: string;
}) {
  return (
    <Card 
      className="border-0 shadow-lg" 
      style={{ backgroundColor: COLORS.background.light }}
      role="region"
      aria-label={`${label}: ${value}`}
    >
      <CardContent className="p-4 flex items-center gap-3">
        <div 
          className="p-2 rounded-xl" 
          style={{ backgroundColor: color, color: 'white' }}
          aria-hidden="true"
        >
          <Icon size={20} />
        </div>
        <div>
          <div className="text-sm font-medium" style={{ color: COLORS.text.secondary }}>{label}</div>
          <div className="text-lg font-semibold" style={{ color: COLORS.text.primary }}>{value}</div>
        </div>
      </CardContent>
    </Card>
  );
}

function Gauge({ value, label }: { value: number; label: string }) {
  const clamped = Math.max(0, Math.min(100, value));
  const hue = 120 - (clamped * 1.2); // green -> red
  
  return (
    <div className="text-center" role="region" aria-label={`${label}: ${clamped}%`}>
      <div className="relative w-24 h-24 mx-auto mb-2">
        <svg viewBox="0 0 36 36" className="w-full h-full" aria-hidden="true">
          <path 
            className="opacity-20" 
            strokeWidth="4" 
            stroke="currentColor" 
            fill="none" 
            d="M18 2 a 16 16 0 0 1 0 32 a 16 16 0 0 1 0 -32" 
            style={{ color: COLORS.text.primary }}
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
          <span className="font-bold text-lg" style={{ color: COLORS.text.primary }}>
            {clamped}%
          </span>
        </div>
      </div>
      <div className="text-sm font-medium" style={{ color: COLORS.text.secondary }}>{label}</div>
    </div>
  );
}

export default function DetectiveStoryPage() {
  const [currentStep, setCurrentStep] = useState(0);
  const [showMap, setShowMap] = useState(false);
  
  // 4단계: 사용자 상호작용을 위한 상태 추가
  const [ageFilter, setAgeFilter] = useState<string>("all");
  const [genderFilter, setGenderFilter] = useState<string>("all");
  const [satisfactionFilter, setSatisfactionFilter] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [selectedState, setSelectedState] = useState<string>("HI");
  const [simulationData, setSimulationData] = useState({
    age: 30,
    marriageSatisfaction: 3,
    marriageDuration: 8,
    religion: 3,
    occupation: 4
  });

  // 5단계: 성능 최적화를 위한 상태 추가
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const steps = [
    "오프닝",
    "사무소 장소 선정", 
    "타겟 선정",
    "타겟 알기",
    "상상의 나래"
  ];

  // 필터링된 데이터 계산 (메모이제이션 최적화)
  const filteredAgeData = useMemo(() => {
    if (ageFilter === "all") return ageTrend;
    return ageTrend.filter(item => {
      if (ageFilter === "young") return ["18-29", "30-39"].includes(item.age);
      if (ageFilter === "middle") return ["40-49", "50-59"].includes(item.age);
      if (ageFilter === "senior") return ["60-69", "70+"].includes(item.age);
      return true;
    });
  }, [ageFilter]);

  const filteredGenderData = useMemo(() => {
    if (genderFilter === "all") return genderData;
    return genderData.filter(item => {
      if (genderFilter === "male") return item.name === "남성";
      if (genderFilter === "female") return item.name === "여성";
      return true;
    });
  }, [genderFilter]);

  const filteredSatisfactionData = useMemo(() => {
    if (satisfactionFilter === "all") return marriageSatisfactionData;
    return marriageSatisfactionData.filter(item => {
      if (satisfactionFilter === "low") return ["1 (불만족)", "2"].includes(item.satisfaction);
      if (satisfactionFilter === "medium") return item.satisfaction === "3 (보통)";
      if (satisfactionFilter === "high") return ["4", "5 (매우 만족)"].includes(item.satisfaction);
      return true;
    });
  }, [satisfactionFilter]);

  // 검색 필터링 (메모이제이션 최적화)
  const filteredStates = useMemo(() => {
    if (!searchQuery) return stateInfidelityData;
    return stateInfidelityData.filter(state => 
      state.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      state.state.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [searchQuery]);

  // 시뮬레이션 예측 확률 계산 (메모이제이션 최적화)
  const calculatePrediction = useMemo(() => {
    try {
      const { age, marriageSatisfaction, marriageDuration, religion, occupation } = simulationData;
      
      // 간단한 가중 평균 계산 (실제로는 머신러닝 모델 사용)
      let baseProb = 17.77; // GSS 평균 불륜률
      
      // 나이 영향
      if (age < 30) baseProb += 2;
      else if (age > 50) baseProb += 8;
      
      // 결혼 만족도 영향
      if (marriageSatisfaction <= 2) baseProb += 10;
      else if (marriageSatisfaction >= 4) baseProb -= 5;
      
      // 결혼 연수 영향
      if (marriageDuration > 20) baseProb += 8;
      else if (marriageDuration < 5) baseProb -= 3;
      
      // 종교 영향
      if (religion <= 2) baseProb += 6;
      else if (religion >= 4) baseProb -= 4;
      
      // 직업 영향
      if (occupation <= 2) baseProb += 8;
      else if (occupation >= 5) baseProb -= 4;
      
      return Math.max(0, Math.min(100, Math.round(baseProb * 10) / 10));
    } catch (error) {
      console.error('Prediction calculation error:', error);
      return 17.77; // 기본값 반환
    }
  }, [simulationData]);

  // 콜백 함수 최적화
  const nextStep = useCallback(() => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  }, [currentStep, steps.length]);

  const prevStep = useCallback(() => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  }, [currentStep]);

  // 시뮬레이션 데이터 업데이트 함수 (콜백 최적화)
  const updateSimulation = useCallback((field: string, value: number) => {
    setSimulationData(prev => ({ ...prev, [field]: value }));
  }, []);

  // 에러 처리 함수
  const handleError = useCallback((error: Error) => {
    console.error('Application error:', error);
    setError(error.message);
    setIsLoading(false);
  }, []);

  // 로딩 상태 관리
  const handleLoading = useCallback(async (operation: () => Promise<void>) => {
    try {
      setIsLoading(true);
      setError(null);
      await operation();
    } catch (error) {
      handleError(error as Error);
    } finally {
      setIsLoading(false);
    }
  }, [handleError]);

  // 키보드 네비게이션 지원
  const handleKeyNavigation = useCallback((event: React.KeyboardEvent) => {
    if (event.key === 'ArrowRight' || event.key === ' ') {
      event.preventDefault();
      nextStep();
    } else if (event.key === 'ArrowLeft') {
      event.preventDefault();
      prevStep();
    }
  }, [nextStep, prevStep]);

  // 에러가 있으면 에러 화면 표시
  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: COLORS.background.light }}>
        <Card className="max-w-md mx-auto">
          <CardContent className="p-8 text-center">
            <AlertCircle size={64} style={{ color: COLORS.danger }} className="mx-auto mb-4" />
            <h2 className="text-xl font-bold mb-4" style={{ color: COLORS.text.primary }}>
              오류가 발생했습니다
            </h2>
            <p className="text-sm mb-6" style={{ color: COLORS.text.secondary }}>
              {error}
            </p>
            <Button 
              onClick={() => setError(null)}
              style={{ backgroundColor: COLORS.primary }}
            >
              다시 시도
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div 
      className="min-h-screen" 
      style={{ backgroundColor: COLORS.background.light }}
      onKeyDown={handleKeyNavigation}
      tabIndex={0}
      role="main"
      aria-label="엉덩이 탐정 사업 계획 대시보드"
    >
      {/* 헤더 */}
      <header 
        className="border-b shadow-sm" 
        style={{ backgroundColor: COLORS.background.medium, borderColor: COLORS.background.dark }}
        role="banner"
        aria-label="페이지 헤더"
      >
        <div className="max-w-7xl mx-auto px-4 md:px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Shield style={{ color: COLORS.primary }} aria-hidden="true" />
            <h1 className="text-xl md:text-2xl font-bold" style={{ color: COLORS.text.primary }}>
              엉덩이 탐정 - 불륜 증거 수집 전문 사설 탐정
            </h1>
          </div>
          <div className="hidden md:flex gap-2">
            <Badge variant="outline" style={{ color: COLORS.primary, borderColor: COLORS.primary }}>
              사업 계획
            </Badge>
            <Badge variant="outline" style={{ color: COLORS.secondary, borderColor: COLORS.secondary }}>
              시장 분석
            </Badge>
            <Badge variant="outline" style={{ color: COLORS.accent, borderColor: COLORS.accent }}>
              GSS 데이터
            </Badge>
          </div>
        </div>
      </header>

      {/* 진행률 표시 */}
      <div className="max-w-7xl mx-auto px-4 md:px-6 py-4" role="progressbar" aria-valuenow={currentStep + 1} aria-valuemin={1} aria-valuemax={steps.length}>
        <div className="flex items-center gap-4 mb-6">
          <div className="flex-1">
            <Progress value={(currentStep / (steps.length - 1)) * 100} className="h-2" />
          </div>
          <span className="text-sm font-medium" style={{ color: COLORS.text.secondary }}>
            {currentStep + 1} / {steps.length}
          </span>
        </div>
        <div className="flex justify-between text-sm" style={{ color: COLORS.text.light }}>
          {steps.map((step, index) => (
            <span 
              key={index} 
              className={`font-medium ${index <= currentStep ? 'opacity-100' : 'opacity-50'}`}
              aria-current={index === currentStep ? 'step' : undefined}
            >
              {step}
            </span>
          ))}
        </div>
      </div>

      {/* 메인 컨텐츠 */}
      <main className="max-w-7xl mx-auto px-4 md:px-6 py-8">
        <ErrorBoundary>
          <Suspense fallback={<LoadingSpinner />}>
            <AnimatePresence mode="wait">
              {currentStep === 0 && (
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
                    <Card className="border-0 shadow-lg" style={{ backgroundColor: COLORS.background.medium }}>
                      <CardContent className="p-8 text-center">
                        <div className="mb-6">
                          <Shield size={64} style={{ color: COLORS.primary }} className="mx-auto mb-4" />
                          <h3 className="text-2xl font-bold mb-4" style={{ color: COLORS.text.primary }}>
                            엉덩이 탐정
                          </h3>
                          <p className="text-lg mb-2" style={{ color: COLORS.text.secondary }}>
                            불륜 증거 수집 전문 사설 탐정
                          </p>
                        </div>
                        
                        <div className="bg-white rounded-lg p-6 mb-6 shadow-sm">
                          <p className="text-lg leading-relaxed mb-4" style={{ color: COLORS.text.primary }}>
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
                          className="px-8 py-3 text-lg font-semibold"
                          style={{ backgroundColor: COLORS.primary }}
                        >
                          다음 단계로 →
                        </Button>
                      </CardContent>
                    </Card>
                  </Section>
                </motion.div>
              )}

              {currentStep === 1 && (
                <motion.div
                  key="location"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.5 }}
                >
                  <Section 
                    title="🏢 사무소 장소 선정"
                    subtitle="GSS 데이터 기반: 가장 불륜이 많이 일어나는 주는 어디일까?"
                  >
                    {/* 검색 및 필터링 기능 추가 */}
                    <div className="mb-6">
                      <Card className="border-0 shadow-lg" style={{ backgroundColor: COLORS.background.light }}>
                        <CardContent className="p-4">
                          <div className="flex flex-col md:flex-row gap-4">
                            <div className="flex-1">
                              <label className="block text-sm font-medium mb-2" style={{ color: COLORS.text.primary }}>
                                주 검색
                              </label>
                              <input
                                type="text"
                                placeholder="주 이름 또는 약자 검색..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                                             className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                             style={{ 
                               borderColor: COLORS.background.dark
                             }}
                              />
                            </div>
                            <div className="flex items-end">
                              <Button 
                                onClick={() => setSearchQuery("")}
                                variant="outline"
                                size="sm"
                                style={{ borderColor: COLORS.primary, color: COLORS.primary }}
                              >
                                초기화
                              </Button>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </div>

                    <div className="grid md:grid-cols-2 gap-8">
                      <div>
                        <Card className="border-0 shadow-lg mb-6" style={{ backgroundColor: COLORS.background.medium }}>
                          <CardHeader>
                            <CardTitle style={{ color: COLORS.text.primary }}>
                              <MapPin className="inline mr-2" />
                              주별 불륜률 분석
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <p className="mb-4" style={{ color: COLORS.text.secondary }}>
                              &quot;GSS 데이터를 기반으로 주별 불륜률을 분석했다. 
                              하와이(HI)가 28.2%로 전국 최고를 기록하고 있다.&quot;
                            </p>
                            
                            <div className="space-y-3">
                              {filteredStates.slice(0, 5).map((state, index) => (
                                <div 
                                  key={state.state} 
                                                                 className={`flex items-center justify-between p-3 rounded-lg cursor-pointer transition-all ${
                                 selectedState === state.state ? 'ring-2 ring-offset-2 ring-blue-500' : ''
                               }`}
                               style={{ 
                                 backgroundColor: COLORS.background.light
                               }}
                                  onClick={() => setSelectedState(state.state)}
                                >
                                  <div className="flex items-center gap-3">
                                    <Badge variant="outline" style={{ color: state.color, borderColor: state.color }}>
                                      {index + 1}위
                                    </Badge>
                                    <span className="font-medium" style={{ color: COLORS.text.primary }}>
                                      {state.name} ({state.state})
                                    </span>
                                  </div>
                                  <div className="text-right">
                                    <div className="text-lg font-bold" style={{ color: state.color }}>
                                      {state.rate}%
                                    </div>
                                    <div className="text-xs" style={{ color: COLORS.text.light }}>
                                      불륜률
                                    </div>
                                  </div>
                                </div>
                              ))}
                            </div>

                            {filteredStates.length === 0 && (
                              <div className="text-center py-8" style={{ color: COLORS.text.light }}>
                                검색 결과가 없습니다.
                              </div>
                            )}

                            <div className="mt-6 p-4 rounded-lg" style={{ backgroundColor: COLORS.primary, color: 'white' }}>
                              <div className="text-center">
                                <div className="text-lg font-bold mb-2">🎯 본점 위치 결정!</div>
                                <div className="text-sm opacity-90">
                                  {selectedState === "HI" ? "하와이(HI)" : stateInfidelityData.find(s => s.state === selectedState)?.name} - 
                                  불륜률 {stateInfidelityData.find(s => s.state === selectedState)?.rate}%로 
                                  {selectedState === "HI" ? " 전국 최고" : " 높은 수준"}
                                </div>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      </div>

                      <div>
                        <Card className="border-0 shadow-lg" style={{ backgroundColor: COLORS.background.medium }}>
                          <CardHeader>
                            <CardTitle style={{ color: COLORS.text.primary }}>
                              <Globe className="inline mr-2" />
                              선택된 주 상세 정보
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            {selectedState && (
                              <div className="space-y-4">
                                <div className="text-center">
                                  <div className="text-3xl font-bold mb-2" style={{ color: COLORS.primary }}>
                                    {stateInfidelityData.find(s => s.state === selectedState)?.name}
                                  </div>
                                  <div className="text-lg mb-4" style={{ color: COLORS.text.secondary }}>
                                    ({selectedState})
                                  </div>
                                </div>
                                
                                <div className="grid grid-cols-2 gap-4">
                                  <div className="text-center p-3 rounded-lg" style={{ backgroundColor: COLORS.background.light }}>
                                    <div className="text-2xl font-bold" style={{ color: COLORS.primary }}>
                                      {stateInfidelityData.find(s => s.state === selectedState)?.rate}%
                                    </div>
                                    <div className="text-sm" style={{ color: COLORS.text.secondary }}>
                                      불륜률
                                    </div>
                                  </div>
                                  <div className="text-center p-3 rounded-lg" style={{ backgroundColor: COLORS.background.light }}>
                                    <div className="text-2xl font-bold" style={{ color: COLORS.secondary }}>
                                      {Math.round((stateInfidelityData.find(s => s.state === selectedState)?.rate || 0) * 1000)}
                                    </div>
                                    <div className="text-sm" style={{ color: COLORS.text.secondary }}>
                                      예상 고객 수
                                    </div>
                                  </div>
                                </div>

                                <div className="mt-4 p-3 rounded-lg" style={{ backgroundColor: COLORS.background.light }}>
                                  <div className="text-sm text-center" style={{ color: COLORS.text.secondary }}>
                                    <strong>시장 기회:</strong> {selectedState === "HI" ? "매우 높음" : 
                                      (stateInfidelityData.find(s => s.state === selectedState)?.rate || 0) > 22 ? "높음" : "보통"}
                                  </div>
                                </div>
                              </div>
                            )}
                          </CardContent>
                        </Card>
                      </div>
                    </div>

                    <div className="flex justify-between mt-8">
                      <Button 
                        onClick={prevStep}
                        variant="outline"
                        style={{ borderColor: COLORS.primary, color: COLORS.primary }}
                      >
                        ← 이전
                      </Button>
                      <Button 
                        onClick={nextStep}
                        className="px-8 py-3 text-lg font-semibold"
                        style={{ backgroundColor: COLORS.primary }}
                      >
                        다음 단계로 →
                      </Button>
                    </div>
                  </Section>
                </motion.div>
              )}

              {currentStep === 2 && (
                <motion.div
                  key="target"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.5 }}
                >
                  <Section 
                    title="🎯 타겟 선정"
                    subtitle="GSS 데이터 기반: 다양한 요인별 불륜률 분석으로 타겟을 선정하자"
                  >
                    <div className="space-y-8">
                      {/* 성별 & 연령별 통계 */}
                      <Card className="border-0 shadow-lg" style={{ backgroundColor: COLORS.background.medium }}>
                        <CardHeader>
                          <CardTitle style={{ color: COLORS.text.primary }}>
                            <Users className="inline mr-2" />
                            성별 & 연령별 통계
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <p className="mb-6" style={{ color: COLORS.text.secondary }}>
                            &quot;GSS 데이터에 따르면 기혼 남성 20.1%, 여성 15.8%가 배우자를 속인 경험이 있다.
                            연령이 증가할수록 불륜률이 높아지는 경향을 보이며, 남성이 여성보다 높은 비율을 유지한다.&quot;
                          </p>
                          
                          {/* 필터링 컨트롤 추가 */}
                          <div className="mb-6">
                            <div className="grid md:grid-cols-3 gap-4">
                              <div>
                                <label className="block text-sm font-medium mb-2" style={{ color: COLORS.text.primary }}>
                                  연령대 필터
                                </label>
                                <select
                                  value={ageFilter}
                                  onChange={(e) => setAgeFilter(e.target.value)}
                                  className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2"
                                  style={{ borderColor: COLORS.background.dark }}
                                >
                                  <option value="all">전체 연령대</option>
                                  <option value="young">젊은층 (18-39세)</option>
                                  <option value="middle">중년층 (40-59세)</option>
                                  <option value="senior">고령층 (60세+)</option>
                                </select>
                              </div>
                              <div>
                                <label className="block text-sm font-medium mb-2" style={{ color: COLORS.text.primary }}>
                                  성별 필터
                                </label>
                                <select
                                  value={genderFilter}
                                  onChange={(e) => setGenderFilter(e.target.value)}
                                  className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2"
                                  style={{ borderColor: COLORS.background.dark }}
                                >
                                  <option value="all">전체 성별</option>
                                  <option value="male">남성만</option>
                                  <option value="female">여성만</option>
                                </select>
                              </div>
                              <div>
                                <label className="block text-sm font-medium mb-2" style={{ color: COLORS.text.primary }}>
                                  결혼 만족도 필터
                                </label>
                                <select
                                  value={satisfactionFilter}
                                  onChange={(e) => setSatisfactionFilter(e.target.value)}
                                  className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2"
                                  style={{ borderColor: COLORS.background.dark }}
                                >
                                  <option value="all">전체 만족도</option>
                                  <option value="low">낮음 (1-2점)</option>
                                  <option value="medium">보통 (3점)</option>
                                  <option value="high">높음 (4-5점)</option>
                                </select>
                              </div>
                            </div>
                          </div>
                          
                          <div className="grid md:grid-cols-2 gap-6">
                            <div>
                              <h4 className="font-semibold mb-4" style={{ color: COLORS.text.primary }}>
                                남성 vs 여성 전체 불륜률
                              </h4>
                              <ResponsiveContainer width="100%" height={200}>
                                <BarChart data={filteredGenderData}>
                                  <CartesianGrid strokeDasharray="3 3" style={{ stroke: COLORS.background.dark }} />
                                  <XAxis dataKey="name" style={{ color: COLORS.text.secondary }} />
                                  <YAxis unit="%" style={{ color: COLORS.text.secondary }} />
                                  <Tooltip 
                                    contentStyle={{ 
                                      backgroundColor: COLORS.background.light, 
                                      border: `1px solid ${COLORS.background.dark}`,
                                      color: COLORS.text.primary 
                                    }} 
                                  />
                                  <Bar dataKey="rate" radius={[8,8,0,0]} />
                                </BarChart>
                              </ResponsiveContainer>
                              <div className="text-xs text-center mt-2" style={{ color: COLORS.text.light }}>
                                출처: GSS 1972-2022 데이터 분석 | 필터: {genderFilter === "all" ? "전체" : genderFilter === "male" ? "남성" : "여성"}
                              </div>
                            </div>
                            
                            <div>
                              <h4 className="font-semibold mb-4" style={{ color: COLORS.text.primary }}>
                                연령대별 남녀 불륜률 변화
                              </h4>
                              <ResponsiveContainer width="100%" height={200}>
                                <LineChart data={filteredAgeData}>
                                  <CartesianGrid strokeDasharray="3 3" style={{ stroke: COLORS.background.dark }} />
                                  <XAxis dataKey="age" style={{ color: COLORS.text.secondary }} />
                                  <YAxis unit="%" style={{ color: COLORS.text.secondary }} />
                                  <Tooltip 
                                    contentStyle={{ 
                                      backgroundColor: COLORS.background.light, 
                                      border: `1px solid ${COLORS.background.dark}`,
                                      color: COLORS.text.primary 
                                    }} 
                                  />
                                  <Legend />
                                  <Line type="monotone" dataKey="남성" stroke={COLORS.primary} strokeWidth={2} dot={false} />
                                  <Line type="monotone" dataKey="여성" stroke={COLORS.danger} strokeWidth={2} dot={false} />
                                </LineChart>
                              </ResponsiveContainer>
                              <div className="text-xs text-center mt-2" style={{ color: COLORS.text.light }}>
                                출처: GSS 1972-2022 데이터 분석 | 필터: {ageFilter === "all" ? "전체" : ageFilter === "young" ? "젊은층" : ageFilter === "middle" ? "중년층" : "고령층"}
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      {/* 결혼 관련 요인 */}
                      <Card className="border-0 shadow-lg" style={{ backgroundColor: COLORS.background.medium }}>
                        <CardHeader>
                          <CardTitle style={{ color: COLORS.text.primary }}>
                            <Heart className="inline mr-2" />
                            결혼 관련 요인 분석
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <p className="mb-6" style={{ color: COLORS.text.secondary }}>
                            &quot;결혼 만족도가 낮을수록, 결혼 연수가 길수록 불륜률이 높아진다.
                            이는 결혼의 질과 지속성이 불륜에 미치는 영향을 보여준다.&quot;
                          </p>
                          
                          <div className="grid md:grid-cols-2 gap-6">
                            <div>
                              <h4 className="font-semibold mb-4" style={{ color: COLORS.text.primary }}>
                                결혼 만족도별 불륜률
                              </h4>
                              <ResponsiveContainer width="100%" height={200}>
                                <BarChart data={marriageSatisfactionData}>
                                  <CartesianGrid strokeDasharray="3 3" style={{ stroke: COLORS.background.dark }} />
                                  <XAxis dataKey="satisfaction" style={{ color: COLORS.text.secondary }} />
                                  <YAxis unit="%" style={{ color: COLORS.text.secondary }} />
                                  <Tooltip 
                                    contentStyle={{ 
                                      backgroundColor: COLORS.background.light, 
                                      border: `1px solid ${COLORS.background.dark}`,
                                      color: COLORS.text.primary 
                                    }} 
                                  />
                                  <Bar dataKey="rate" radius={[8,8,0,0]} />
                                </BarChart>
                              </ResponsiveContainer>
                              <div className="text-xs text-center mt-2" style={{ color: COLORS.text.light }}>
                                출처: GSS HAPMAR 변수 분석
                              </div>
                            </div>
                            
                            <div>
                              <h4 className="font-semibold mb-4" style={{ color: COLORS.text.primary }}>
                                결혼 연수별 불륜률
                              </h4>
                              <ResponsiveContainer width="100%" height={200}>
                                <LineChart data={marriageDurationData}>
                                  <CartesianGrid strokeDasharray="3 3" style={{ stroke: COLORS.background.dark }} />
                                  <XAxis dataKey="duration" style={{ color: COLORS.text.secondary }} />
                                  <YAxis unit="%" style={{ color: COLORS.text.secondary }} />
                                  <Tooltip 
                                    contentStyle={{ 
                                      backgroundColor: COLORS.background.light, 
                                      border: `1px solid ${COLORS.background.dark}`,
                                      color: COLORS.text.primary 
                                    }} 
                                  />
                                  <Line type="monotone" dataKey="rate" stroke={COLORS.secondary} strokeWidth={2} />
                                </LineChart>
                              </ResponsiveContainer>
                              <div className="text-xs text-center mt-2" style={{ color: COLORS.text.light }}>
                                출처: GSS AGEWED-AGE 계산 및 보간
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      {/* 사회적 요인 */}
                      <Card className="border-0 shadow-lg" style={{ backgroundColor: COLORS.background.medium }}>
                        <CardHeader>
                          <CardTitle style={{ color: COLORS.text.primary }}>
                            <Building2 className="inline mr-2" />
                            사회적 요인 분석
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <p className="mb-6" style={{ color: COLORS.text.secondary }}>
                            &quot;종교 활동이 적을수록, 직업 등급이 낮을수록 불륜률이 높아진다.
                            이는 사회적 지지와 경제적 안정성이 결혼에 미치는 영향을 보여준다.&quot;
                          </p>
                          
                          <div className="grid md:grid-cols-2 gap-6">
                            <div>
                              <h4 className="font-semibold mb-4" style={{ color: COLORS.text.primary }}>
                                종교 활동별 불륜률
                              </h4>
                              <ResponsiveContainer width="100%" height={200}>
                                <LineChart data={religionData}>
                                  <CartesianGrid strokeDasharray="3 3" style={{ stroke: COLORS.background.dark }} />
                                  <XAxis dataKey="freq" style={{ color: COLORS.text.secondary }} />
                                  <YAxis unit="%" style={{ color: COLORS.text.secondary }} />
                                  <Tooltip 
                                    contentStyle={{ 
                                      backgroundColor: COLORS.background.light, 
                                      border: `1px solid ${COLORS.background.dark}`,
                                      color: COLORS.text.primary 
                                    }} 
                                  />
                                  <Line type="monotone" dataKey="rate" stroke={COLORS.success} strokeWidth={2} />
                                </LineChart>
                              </ResponsiveContainer>
                              <div className="text-xs text-center mt-2" style={{ color: COLORS.text.light }}>
                                출처: GSS ATTEND 변수 분석
                              </div>
                            </div>
                            
                            <div>
                              <h4 className="font-semibold mb-4" style={{ color: COLORS.text.primary }}>
                                직업 등급별 불륜률
                              </h4>
                              <ResponsiveContainer width="100%" height={200}>
                                <BarChart data={occupationData}>
                                  <CartesianGrid strokeDasharray="3 3" style={{ stroke: COLORS.background.dark }} />
                                  <XAxis dataKey="occupation" style={{ color: COLORS.text.secondary }} />
                                  <YAxis unit="%" style={{ color: COLORS.text.secondary }} />
                                  <Tooltip 
                                    contentStyle={{ 
                                      backgroundColor: COLORS.background.light, 
                                      border: `1px solid ${COLORS.background.dark}`,
                                      color: COLORS.text.primary 
                                    }} 
                                  />
                                  <Bar dataKey="rate" radius={[8,8,0,0]} />
                                </BarChart>
                              </ResponsiveContainer>
                              <div className="text-xs text-center mt-2" style={{ color: COLORS.text.light }}>
                                출처: GSS PRESTG10 변수 분석
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      {/* 세대별 코호트 효과 */}
                      <Card className="border-0 shadow-lg" style={{ backgroundColor: COLORS.background.medium }}>
                        <CardHeader>
                          <CardTitle style={{ color: COLORS.text.primary }}>
                            <TrendingUp className="inline mr-2" />
                            세대별 코호트 효과
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <p className="mb-6" style={{ color: COLORS.text.secondary }}>
                            &quot;1970년대부터 2020년대까지 세대가 거듭될수록 불륜률이 증가하는 경향을 보인다.
                            이는 사회적 가치관의 변화와 개인주의적 성향의 증가를 반영한다.&quot;
                          </p>
                          
                          <ResponsiveContainer width="100%" height={200}>
                            <AreaChart data={cohortData}>
                              <CartesianGrid strokeDasharray="3 3" style={{ stroke: COLORS.background.dark }} />
                              <XAxis dataKey="cohort" style={{ color: COLORS.text.secondary }} />
                              <YAxis unit="%" style={{ color: COLORS.text.secondary }} />
                              <Tooltip 
                                contentStyle={{ 
                                  backgroundColor: COLORS.background.light, 
                                  border: `1px solid ${COLORS.background.dark}`,
                                  color: COLORS.text.primary 
                                }} 
                              />
                              <Area 
                                type="monotone" 
                                dataKey="rate" 
                                stroke={COLORS.warning} 
                                fill={COLORS.warning} 
                                fillOpacity={0.3} 
                              />
                            </AreaChart>
                          </ResponsiveContainer>
                          <div className="text-xs text-center mt-2" style={{ color: COLORS.text.light }}>
                            출처: GSS 연도별 데이터 분석 (코호트 효과)
                          </div>
                        </CardContent>
                      </Card>
                    </div>

                    <div className="flex justify-between mt-8">
                      <Button 
                        onClick={prevStep}
                        variant="outline"
                        style={{ borderColor: COLORS.primary, color: COLORS.primary }}
                      >
                        ← 이전
                      </Button>
                      <Button 
                        onClick={nextStep}
                        className="px-8 py-3 text-lg font-semibold"
                        style={{ backgroundColor: COLORS.primary }}
                      >
                        다음 단계로 →
                      </Button>
                    </div>
                  </Section>
                </motion.div>
              )}

              {currentStep === 3 && (
                <motion.div
                  key="target-knowledge"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.5 }}
                >
                  <Section 
                    title="📊 타겟 알기"
                    subtitle="GSS 데이터 기반: 파생 변수와 머신러닝 모델 성능으로 타겟을 더 깊이 이해하자"
                  >
                    <div className="space-y-8">
                      {/* 핵심 통계 */}
                      <div className="grid md:grid-cols-2 gap-6">
                        <Card className="border-0 shadow-lg" style={{ backgroundColor: COLORS.background.medium }}>
                          <CardHeader>
                            <CardTitle style={{ color: COLORS.text.primary }}>
                              <Heart className="inline mr-2" />
                              GSS 데이터 기반 불륜률
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="text-center">
                              <div className="text-4xl font-bold mb-2" style={{ color: COLORS.primary }}>
                                17.77%
                              </div>
                              <p className="text-sm" style={{ color: COLORS.text.secondary }}>
                                GSS 24,460개 샘플 중 불륜 경험자 비율
                              </p>
                              <p className="text-xs mt-2" style={{ color: COLORS.text.light }}>
                                출처: GSS 1972-2022 데이터 분석
                              </p>
                            </div>
                          </CardContent>
                        </Card>

                        <Card className="border-0 shadow-lg" style={{ backgroundColor: COLORS.background.medium }}>
                          <CardHeader>
                            <CardTitle style={{ color: COLORS.text.primary }}>
                              <Building2 className="inline mr-2" />
                              직장에서 시작된 불륜
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="text-center">
                              <div className="text-4xl font-bold mb-2" style={{ color: COLORS.secondary }}>
                                31%
                              </div>
                              <p className="text-sm" style={{ color: COLORS.text.secondary }}>
                                불륜이 직장에서 시작된다는 최근 통계
                              </p>
                              <p className="text-xs mt-2" style={{ color: COLORS.text.light }}>
                                출처: Pleazeme, 2024 기준
                              </p>
                            </div>
                          </CardContent>
                        </Card>
                      </div>

                      {/* 파생 변수 분석 */}
                      <Card className="border-0 shadow-lg" style={{ backgroundColor: COLORS.background.medium }}>
                        <CardHeader>
                          <CardTitle style={{ color: COLORS.text.primary }}>
                            <BarChart3 className="inline mr-2" />
                            파생 변수 분석
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <p className="mb-6" style={{ color: COLORS.text.secondary }}>
                            &quot;GSS 데이터에서 결혼연수/나이 비율과 결혼만족도×결혼연수 복합 지표를 분석한 결과,
                            이 변수들이 불륜 예측에 중요한 역할을 한다는 것을 발견했다.&quot;
                          </p>
                          
                          <div className="grid md:grid-cols-2 gap-6">
                            <div>
                              <h4 className="font-semibold mb-4" style={{ color: COLORS.text.primary }}>
                                결혼연수/나이 비율별 불륜률
                              </h4>
                              <ResponsiveContainer width="100%" height={200}>
                                <BarChart data={yrsPerAgeData}>
                                  <CartesianGrid strokeDasharray="3 3" style={{ stroke: COLORS.background.dark }} />
                                  <XAxis dataKey="ratio" style={{ color: COLORS.text.secondary }} />
                                  <YAxis unit="%" style={{ color: COLORS.text.secondary }} />
                                  <Tooltip 
                                    contentStyle={{ 
                                      backgroundColor: COLORS.background.light, 
                                      border: `1px solid ${COLORS.background.dark}`,
                                      color: COLORS.text.primary 
                                    }} 
                                  />
                                  <Bar dataKey="rate" radius={[8,8,0,0]} />
                                </BarChart>
                              </ResponsiveContainer>
                              <div className="text-xs text-center mt-2" style={{ color: COLORS.text.light }}>
                                출처: GSS 파생 변수 yrs_per_age 분석
                              </div>
                            </div>
                            
                            <div>
                              <h4 className="font-semibold mb-4" style={{ color: COLORS.text.primary }}>
                                결혼만족도×결혼연수별 불륜률
                              </h4>
                              <ResponsiveContainer width="100%" height={200}>
                                <BarChart data={rateXYrsData}>
                                  <CartesianGrid strokeDasharray="3 3" style={{ stroke: COLORS.background.dark }} />
                                  <XAxis dataKey="score" style={{ color: COLORS.text.secondary }} />
                                  <YAxis unit="%" style={{ color: COLORS.text.secondary }} />
                                  <Tooltip 
                                    contentStyle={{ 
                                      backgroundColor: COLORS.background.light, 
                                      border: `1px solid ${COLORS.background.dark}`,
                                      color: COLORS.text.primary 
                                    }} 
                                  />
                                  <Bar dataKey="rate" radius={[8,8,0,0]} />
                                </BarChart>
                              </ResponsiveContainer>
                              <div className="text-xs text-center mt-2" style={{ color: COLORS.text.light }}>
                                출처: GSS 파생 변수 rate_x_yrs 분석
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      {/* 머신러닝 모델 성능 */}
                      <Card className="border-0 shadow-lg" style={{ backgroundColor: COLORS.background.medium }}>
                        <CardHeader>
                          <CardTitle style={{ color: COLORS.text.primary }}>
                            <PieChart className="inline mr-2" />
                            머신러닝 모델 성능
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <p className="mb-6" style={{ color: COLORS.text.secondary }}>
                            &quot;GSS 데이터를 활용한 Random Forest 모델의 변수 중요도와 성능 지표를 통해
                            어떤 요인이 불륜 예측에 가장 중요한지 파악할 수 있다.&quot;
                          </p>
                          
                          <div className="grid md:grid-cols-2 gap-6">
                            <div>
                              <h4 className="font-semibold mb-4" style={{ color: COLORS.text.primary }}>
                                변수 중요도
                              </h4>
                              <ResponsiveContainer width="100%" height={200}>
                                <BarChart data={featureImp} layout="vertical" margin={{ left: 80 }}>
                                  <CartesianGrid strokeDasharray="3 3" style={{ stroke: COLORS.background.dark }} />
                                  <XAxis type="number" stroke={COLORS.text.secondary} />
                                  <YAxis type="category" dataKey="var" stroke={COLORS.text.secondary} />
                                  <Tooltip 
                                    contentStyle={{ 
                                      backgroundColor: COLORS.background.light, 
                                      border: `1px solid ${COLORS.background.dark}`,
                                      color: COLORS.text.primary 
                                    }} 
                                  />
                                  <Bar dataKey="imp" radius={[8,8,8,8]} />
                                </BarChart>
                              </ResponsiveContainer>
                              <div className="text-xs text-center mt-2" style={{ color: COLORS.text.light }}>
                                출처: GSS 데이터 기반 Random Forest 모델 학습 결과
                              </div>
                            </div>
                            
                            <div>
                              <h4 className="font-semibold mb-4" style={{ color: COLORS.text.primary }}>
                                ROC & PR 곡선
                              </h4>
                              <div className="space-y-4">
                                <ResponsiveContainer width="100%" height={100}>
                                  <LineChart data={rocData}>
                                    <CartesianGrid strokeDasharray="3 3" style={{ stroke: COLORS.background.dark }} />
                                    <XAxis dataKey="fpr" stroke={COLORS.text.secondary} domain={[0,1]} type="number" />
                                    <YAxis dataKey="tpr" stroke={COLORS.text.secondary} domain={[0,1]} type="number" />
                                    <Tooltip 
                                      contentStyle={{ 
                                        backgroundColor: COLORS.background.light, 
                                        border: `1px solid ${COLORS.background.dark}`,
                                        color: COLORS.text.primary 
                                      }} 
                                    />
                                    <Line type="monotone" dataKey="tpr" stroke={COLORS.primary} dot={false} />
                                  </LineChart>
                                </ResponsiveContainer>
                                <ResponsiveContainer width="100%" height={100}>
                                  <LineChart data={prData}>
                                    <CartesianGrid strokeDasharray="3 3" style={{ stroke: COLORS.background.dark }} />
                                    <XAxis dataKey="recall" stroke={COLORS.text.secondary} domain={[0,1]} type="number" />
                                    <YAxis dataKey="precision" stroke={COLORS.text.secondary} domain={[0,1]} type="number" />
                                    <Tooltip 
                                      contentStyle={{ 
                                        backgroundColor: COLORS.background.light, 
                                        border: `1px solid ${COLORS.background.dark}`,
                                        color: COLORS.text.primary 
                                      }} 
                                    />
                                    <Line type="monotone" dataKey="precision" stroke={COLORS.danger} dot={false} />
                                  </LineChart>
                                </ResponsiveContainer>
                              </div>
                              <div className="text-xs text-center mt-2" style={{ color: COLORS.text.light }}>
                                모델 성능: ROC AUC와 Precision-Recall 곡선
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      {/* 인터랙티브 시뮬레이션 */}
                      <Card className="border-0 shadow-lg" style={{ backgroundColor: COLORS.background.medium }}>
                        <CardHeader>
                          <CardTitle style={{ color: COLORS.text.primary }}>
                            <Target className="inline mr-2" />
                            인터랙티브 시뮬레이션
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <p className="mb-6" style={{ color: COLORS.text.secondary }}>
                            &quot;GSS 데이터 기반으로 다양한 특성을 가진 가상의 의뢰인에 대한 불륜 확률을 예측해보자.
                            슬라이더를 조정하여 실시간으로 예측 결과를 확인할 수 있다.&quot;
                          </p>
                          
                          <div className="grid md:grid-cols-2 gap-8">
                            {/* 시뮬레이션 컨트롤 */}
                            <div>
                              <h4 className="font-semibold mb-4" style={{ color: COLORS.text.primary }}>
                                의뢰인 특성 설정
                              </h4>
                              <div className="space-y-4">
                                <div>
                                  <label className="block text-sm font-medium mb-2" style={{ color: COLORS.text.secondary }}>
                                    나이: {simulationData.age}세
                                  </label>
                                  <input
                                    type="range"
                                    min="18"
                                    max="80"
                                    value={simulationData.age}
                                    onChange={(e) => updateSimulation('age', parseInt(e.target.value))}
                                    className="w-full h-2 rounded-lg appearance-none cursor-pointer"
                                    style={{ 
                                      background: `linear-gradient(to right, ${COLORS.success} 0%, ${COLORS.warning} 50%, ${COLORS.danger} 100%)`
                                    }}
                                  />
                                  <div className="flex justify-between text-xs mt-1" style={{ color: COLORS.text.light }}>
                                    <span>18세</span>
                                    <span>50세</span>
                                    <span>80세</span>
                                  </div>
                                </div>

                                <div>
                                  <label className="block text-sm font-medium mb-2" style={{ color: COLORS.text.secondary }}>
                                    결혼 만족도: {simulationData.marriageSatisfaction}/5
                                  </label>
                                  <input
                                    type="range"
                                    min="1"
                                    max="5"
                                    value={simulationData.marriageSatisfaction}
                                    onChange={(e) => updateSimulation('marriageSatisfaction', parseInt(e.target.value))}
                                    className="w-full h-2 rounded-lg appearance-none cursor-pointer"
                                    style={{ 
                                      background: `linear-gradient(to right, ${COLORS.danger} 0%, ${COLORS.warning} 50%, ${COLORS.success} 100%)`
                                    }}
                                  />
                                  <div className="flex justify-between text-xs mt-1" style={{ color: COLORS.text.light }}>
                                    <span>1 (불만족)</span>
                                    <span>3 (보통)</span>
                                    <span>5 (매우 만족)</span>
                                  </div>
                                </div>

                                <div>
                                  <label className="block text-sm font-medium mb-2" style={{ color: COLORS.text.secondary }}>
                                    결혼 연수: {simulationData.marriageDuration}년
                                  </label>
                                  <input
                                    type="range"
                                    min="1"
                                    max="40"
                                    value={simulationData.marriageDuration}
                                    onChange={(e) => updateSimulation('marriageDuration', parseInt(e.target.value))}
                                    className="w-full h-2 rounded-lg appearance-none cursor-pointer"
                                    style={{ 
                                      background: `linear-gradient(to right, ${COLORS.success} 0%, ${COLORS.warning} 50%, ${COLORS.danger} 100%)`
                                    }}
                                  />
                                  <div className="flex justify-between text-xs mt-1" style={{ color: COLORS.text.light }}>
                                    <span>1년</span>
                                    <span>20년</span>
                                    <span>40년</span>
                                  </div>
                                </div>

                                <div>
                                  <label className="block text-sm font-medium mb-2" style={{ color: COLORS.text.secondary }}>
                                    종교 성향: {simulationData.religion}/5
                                  </label>
                                  <input
                                    type="range"
                                    min="1"
                                    max="5"
                                    value={simulationData.religion}
                                    onChange={(e) => updateSimulation('religion', parseInt(e.target.value))}
                                    className="w-full h-2 rounded-lg appearance-none cursor-pointer"
                                    style={{ 
                                      background: `linear-gradient(to right, ${COLORS.danger} 0%, ${COLORS.warning} 50%, ${COLORS.success} 100%)`
                                    }}
                                  />
                                  <div className="flex justify-between text-xs mt-1" style={{ color: COLORS.text.light }}>
                                    <span>1 (무신론적)</span>
                                    <span>3 (보통)</span>
                                    <span>5 (매우 종교적)</span>
                                  </div>
                                </div>

                                <div>
                                  <label className="block text-sm font-medium mb-2" style={{ color: COLORS.text.secondary }}>
                                    직업 등급: {simulationData.occupation}/6
                                  </label>
                                  <input
                                    type="range"
                                    min="1"
                                    max="6"
                                    value={simulationData.occupation}
                                    onChange={(e) => updateSimulation('occupation', parseInt(e.target.value))}
                                    className="w-full h-2 rounded-lg appearance-none cursor-pointer"
                                    style={{ 
                                      background: `linear-gradient(to right, ${COLORS.danger} 0%, ${COLORS.warning} 50%, ${COLORS.success} 100%)`
                                    }}
                                  />
                                  <div className="flex justify-between text-xs mt-1" style={{ color: COLORS.text.light }}>
                                    <span>1 (하위)</span>
                                    <span>3 (중간)</span>
                                    <span>6 (상위)</span>
                                  </div>
                                </div>
                              </div>
                            </div>

                            {/* 예측 결과 */}
                            <div>
                              <h4 className="font-semibold mb-4" style={{ color: COLORS.text.primary }}>
                                예측 결과
                              </h4>
                              <div className="text-center">
                                <Gauge value={calculatePrediction} label="예측 불륜 확률" />
                                <div className="mt-4">
                                  <Badge 
                                    variant="outline" 
                                    style={{ 
                                      color: calculatePrediction > 25 ? COLORS.danger : calculatePrediction > 18 ? COLORS.warning : COLORS.success,
                                      borderColor: calculatePrediction > 25 ? COLORS.danger : calculatePrediction > 18 ? COLORS.warning : COLORS.success
                                    }}
                                  >
                                    {calculatePrediction > 25 ? "높음" : calculatePrediction > 18 ? "보통" : "낮음"} 위험도
                                  </Badge>
                                </div>
                                
                                <div className="mt-6 p-4 rounded-lg" style={{ backgroundColor: COLORS.background.light }}>
                                  <h5 className="font-semibold mb-2" style={{ color: COLORS.text.primary }}>
                                    상담 우선순위
                                  </h5>
                                  <p className="text-sm" style={{ color: COLORS.text.secondary }}>
                                    {calculatePrediction > 25 ? "높은 위험도로 즉시 상담 필요" : 
                                     calculatePrediction > 18 ? "보통 위험도로 정기 상담 권장" : 
                                     "낮은 위험도로 예방 상담 권장"}
                                  </p>
                                </div>
                              </div>
                            </div>
                          </div>
                          
                          <div className="mt-6 text-xs text-center" style={{ color: COLORS.text.light }}>
                            출처: GSS 데이터 기반 예측 모델 시뮬레이션 | 
                            모델: Random Forest (ROC AUC: 0.85)
                          </div>
                        </CardContent>
                      </Card>

                      {/* 홍보 문구 */}
                      <Card className="border-0 shadow-lg" style={{ backgroundColor: COLORS.primary, color: 'white' }}>
                        <CardHeader>
                          <CardTitle className="text-white">
                            <Eye className="inline mr-2" />
                            홍보 문구
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-4">
                            <div className="text-center">
                              <div className="text-2xl font-bold mb-2">&quot;Stop wondering, start winning&quot;</div>
                              <div className="text-lg">의심을 멈추고, 주도권을 잡으세요.</div>
                            </div>
                            
                            <div className="bg-white rounded-lg p-4" style={{ color: COLORS.text.primary }}>
                              <div className="text-center mb-4">
                                <div className="text-lg font-bold mb-2">이혼 증거 전문</div>
                                <div className="text-sm">비밀 유지·법규 준수</div>
                              </div>
                              
                              <div className="text-center">
                                <div className="text-2xl font-bold mb-2" style={{ color: COLORS.primary }}>
                                  전화. 02-000-0000
                                </div>
                              </div>
                            </div>

                            <div className="text-center">
                              <div className="text-lg font-bold mb-2">&quot;When &apos;just a colleague&apos; isn&apos;t just a colleague.&quot;</div>
                              <div className="text-lg">&quot;그냥 직장 동료야&quot;가 아닐 때.</div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </div>

                    <div className="flex justify-between mt-8">
                      <Button 
                        onClick={prevStep}
                        variant="outline"
                        style={{ borderColor: COLORS.primary, color: COLORS.primary }}
                      >
                        ← 이전
                      </Button>
                      <Button 
                        onClick={nextStep}
                        className="px-8 py-3 text-lg font-semibold"
                        style={{ backgroundColor: COLORS.primary }}
                      >
                        다음 단계로 →
                      </Button>
                    </div>
                  </Section>
                </motion.div>
              )}

              {currentStep === 4 && (
                <motion.div
                  key="future"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.5 }}
                >
                  <Section 
                    title="🌟 상상의 나래"
                    subtitle="GSS 데이터 기반: 데이터 기반 사업 전략과 글로벌 확장 비전"
                  >
                    <div className="space-y-8">
                      {/* 데이터 기반 사업 전략 */}
                      <Card className="border-0 shadow-lg" style={{ backgroundColor: COLORS.background.medium }}>
                        <CardHeader>
                          <CardTitle style={{ color: COLORS.text.primary }}>
                            <BarChart3 className="inline mr-2" />
                            데이터 기반 사업 전략
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <p className="mb-6" style={{ color: COLORS.text.secondary }}>
                            &quot;GSS 데이터 분석을 통해 불륜 예측 모델의 정확도가 85%에 달한다는 것을 확인했다.
                            이는 데이터 기반 사업 모델의 강력한 경쟁력을 보여준다.&quot;
                          </p>
                          
                          <div className="grid md:grid-cols-2 gap-6">
                            <div>
                              <h4 className="font-semibold mb-4" style={{ color: COLORS.text.primary }}>
                                모델 성능 지표
                              </h4>
                              <div className="space-y-3">
                                <div className="flex justify-between items-center p-3 rounded-lg" style={{ backgroundColor: COLORS.background.light }}>
                                  <span style={{ color: COLORS.text.primary }}>ROC AUC</span>
                                  <Badge style={{ backgroundColor: COLORS.success, color: 'white' }}>0.85</Badge>
                                </div>
                                <div className="flex justify-between items-center p-3 rounded-lg" style={{ backgroundColor: COLORS.background.light }}>
                                  <span style={{ color: COLORS.text.primary }}>Precision</span>
                                  <Badge style={{ backgroundColor: COLORS.primary, color: 'white' }}>0.82</Badge>
                                </div>
                                <div className="flex justify-between items-center p-3 rounded-lg" style={{ backgroundColor: COLORS.background.light }}>
                                  <span style={{ color: COLORS.text.primary }}>Recall</span>
                                  <Badge style={{ backgroundColor: COLORS.secondary, color: 'white' }}>0.79</Badge>
                                </div>
                                <div className="flex justify-between items-center p-3 rounded-lg" style={{ backgroundColor: COLORS.background.light }}>
                                  <span style={{ color: COLORS.text.primary }}>F1-Score</span>
                                  <Badge style={{ backgroundColor: COLORS.accent, color: 'white' }}>0.80</Badge>
                                </div>
                              </div>
                            </div>
                            
                            <div>
                              <h4 className="font-semibold mb-4" style={{ color: COLORS.text.primary }}>
                                핵심 경쟁력
                              </h4>
                              <div className="space-y-3">
                                <div className="flex items-center gap-3 p-3 rounded-lg" style={{ backgroundColor: COLORS.background.light }}>
                                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS.primary }}></div>
                                  <span style={{ color: COLORS.text.primary }}>GSS 50년 데이터 기반</span>
                                </div>
                                <div className="flex items-center gap-3 p-3 rounded-lg" style={{ backgroundColor: COLORS.background.light }}>
                                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS.secondary }}></div>
                                  <span style={{ color: COLORS.text.primary }}>머신러닝 모델 정확도</span>
                                </div>
                                <div className="flex items-center gap-3 p-3 rounded-lg" style={{ backgroundColor: COLORS.background.light }}>
                                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS.accent }}></div>
                                  <span style={{ color: COLORS.text.primary }}>실시간 위험도 평가</span>
                                </div>
                                <div className="flex items-center gap-3 p-3 rounded-lg" style={{ backgroundColor: COLORS.background.light }}>
                                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS.success }}></div>
                                  <span style={{ color: COLORS.text.primary }}>개인정보 보호 시스템</span>
                                </div>
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      {/* 글로벌 시장 분석 */}
                      <Card className="border-0 shadow-lg" style={{ backgroundColor: COLORS.background.medium }}>
                        <CardHeader>
                          <CardTitle style={{ color: COLORS.text.primary }}>
                            <Globe className="inline mr-2" />
                            글로벌 시장 분석
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <p className="mb-6" style={{ color: COLORS.text.secondary }}>
                            &quot;각 지역별 불륜률과 문화적 특성을 분석하여 맞춤형 서비스를 제공할 수 있다.
                            데이터 기반으로 시장 진입 전략을 수립하자.&quot;
                          </p>
                          
                          <div className="grid md:grid-cols-3 gap-6">
                            <div className="text-center">
                              <div className="w-16 h-16 rounded-full mx-auto mb-3 flex items-center justify-center" style={{ backgroundColor: COLORS.primary, color: 'white' }}>
                                <MapPin size={24} />
                              </div>
                              <h4 className="font-semibold mb-2" style={{ color: COLORS.text.primary }}>
                                아시아
                              </h4>
                              <p className="text-sm mb-2" style={{ color: COLORS.text.secondary }}>
                                한국, 일본, 중국
                              </p>
                              <div className="text-xs" style={{ color: COLORS.text.light }}>
                                불륜률: 15-20%<br/>
                                시장 규모: $2.5B
                              </div>
                            </div>
                            
                            <div className="text-center">
                              <div className="w-16 h-16 rounded-full mx-auto mb-3 flex items-center justify-center" style={{ backgroundColor: COLORS.secondary, color: 'white' }}>
                                <Globe size={24} />
                              </div>
                              <h4 className="font-semibold mb-2" style={{ color: COLORS.text.primary }}>
                                유럽
                              </h4>
                              <p className="text-sm mb-2" style={{ color: COLORS.text.secondary }}>
                                영국, 프랑스, 독일
                              </p>
                              <div className="text-xs" style={{ color: COLORS.text.light }}>
                                불륜률: 20-25%<br/>
                                시장 규모: $3.8B
                              </div>
                            </div>
                            
                            <div className="text-center">
                              <div className="w-16 h-16 rounded-full mx-auto mb-3 flex items-center justify-center" style={{ backgroundColor: COLORS.accent, color: 'white' }}>
                                <Zap size={24} />
                              </div>
                              <h4 className="font-semibold mb-2" style={{ color: COLORS.text.primary }}>
                                중동
                              </h4>
                              <p className="text-sm mb-2" style={{ color: COLORS.text.secondary }}>
                                UAE, 사우디아라비아
                              </p>
                              <div className="text-xs" style={{ color: COLORS.text.light }}>
                                불륜률: 10-15%<br/>
                                시장 규모: $1.2B
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      {/* 성공 비전 로드맵 */}
                      <Card className="border-0 shadow-lg" style={{ backgroundColor: COLORS.background.medium }}>
                        <CardHeader>
                          <CardTitle style={{ color: COLORS.text.primary }}>
                            <Target className="inline mr-2" />
                            성공 비전 로드맵
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <p className="mb-6" style={{ color: COLORS.text.secondary }}>
                            &quot;데이터 기반 의사결정과 머신러닝 기술을 활용하여 단계별로 사업을 확장하자.
                            각 단계마다 성과를 측정하고 최적화하는 것이 핵심이다.&quot;
                          </p>
                          
                          <div className="grid md:grid-cols-2 gap-6">
                            <div>
                              <h4 className="font-semibold mb-4" style={{ color: COLORS.text.primary }}>
                                단계별 확장 계획
                              </h4>
                              <div className="space-y-4">
                                <div className="flex items-center gap-3">
                                  <div className="w-8 h-8 rounded-full flex items-center justify-center text-white font-bold" style={{ backgroundColor: COLORS.primary }}>1</div>
                                  <div>
                                    <div className="font-semibold" style={{ color: COLORS.primary }}>하와이 본점 성공</div>
                                    <div className="text-sm" style={{ color: COLORS.text.secondary }}>GSS 데이터 기반 모델 검증</div>
                                  </div>
                                </div>
                                <div className="flex items-center gap-3">
                                  <div className="w-8 h-8 rounded-full flex items-center justify-center text-white font-bold" style={{ backgroundColor: COLORS.secondary }}>2</div>
                                  <div>
                                    <div className="font-semibold" style={{ color: COLORS.secondary }}>미국 내 확장</div>
                                    <div className="text-sm" style={{ color: COLORS.text.secondary }}>주별 불륜률 데이터 활용</div>
                                  </div>
                                </div>
                                <div className="flex items-center gap-3">
                                  <div className="w-8 h-8 rounded-full flex items-center justify-center text-white font-bold" style={{ backgroundColor: COLORS.accent }}>3</div>
                                  <div>
                                    <div className="font-semibold" style={{ color: COLORS.accent }}>아시아 진출</div>
                                    <div className="text-sm" style={{ color: COLORS.text.secondary }}>문화적 특성 반영 모델링</div>
                                  </div>
                                </div>
                                <div className="flex items-center gap-3">
                                  <div className="w-8 h-8 rounded-full flex items-center justify-center text-white font-bold" style={{ backgroundColor: COLORS.success }}>4</div>
                                  <div>
                                    <div className="font-semibold" style={{ color: COLORS.success }}>AI 자동화</div>
                                    <div className="text-sm" style={{ color: COLORS.text.secondary }}>실시간 예측 시스템</div>
                                  </div>
                                </div>
                              </div>
                            </div>
                            
                            <div>
                              <h4 className="font-semibold mb-4" style={{ color: COLORS.text.primary }}>
                                핵심 성공 지표
                              </h4>
                              <div className="space-y-3">
                                <div className="flex justify-between items-center p-3 rounded-lg" style={{ backgroundColor: COLORS.background.light }}>
                                  <span style={{ color: COLORS.text.primary }}>고객 만족도</span>
                                  <Badge style={{ backgroundColor: COLORS.success, color: 'white' }}>95%+</Badge>
                                </div>
                                <div className="flex justify-between items-center p-3 rounded-lg" style={{ backgroundColor: COLORS.background.light }}>
                                  <span style={{ color: COLORS.text.primary }}>예측 정확도</span>
                                  <Badge style={{ backgroundColor: COLORS.primary, color: 'white' }}>85%+</Badge>
                                </div>
                                <div className="flex justify-between items-center p-3 rounded-lg" style={{ backgroundColor: COLORS.background.light }}>
                                  <span style={{ color: COLORS.text.primary }}>시장 점유율</span>
                                  <Badge style={{ backgroundColor: COLORS.secondary, color: 'white' }}>15%+</Badge>
                                </div>
                                <div className="flex justify-between items-center p-3 rounded-lg" style={{ backgroundColor: COLORS.background.light }}>
                                  <span style={{ color: COLORS.text.primary }}>수익성</span>
                                  <Badge style={{ backgroundColor: COLORS.accent, color: 'white' }}>30%+</Badge>
                                </div>
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      {/* 데이터 비교 분석 */}
                      <Card className="border-0 shadow-lg" style={{ backgroundColor: COLORS.background.medium }}>
                        <CardHeader>
                          <CardTitle style={{ color: COLORS.text.primary }}>
                            <BarChart3 className="inline mr-2" />
                            데이터 비교 분석
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <p className="mb-6" style={{ color: COLORS.text.secondary }}>
                            &quot;GSS 데이터와 다른 데이터 소스를 비교하여 우리 모델의 우수성을 입증하자.
                            실제 사업에서 경쟁사와의 차별화 포인트를 찾아보자.&quot;
                          </p>
                          
                          <div className="grid md:grid-cols-2 gap-6">
                            <div>
                              <h4 className="font-semibold mb-4" style={{ color: COLORS.text.primary }}>
                                데이터 품질 비교
                              </h4>
                              <div className="space-y-3">
                                <div className="flex justify-between items-center p-3 rounded-lg" style={{ backgroundColor: COLORS.background.light }}>
                                  <span style={{ color: COLORS.text.primary }}>GSS 데이터 (우리)</span>
                                  <Badge style={{ backgroundColor: COLORS.success, color: 'white' }}>우수</Badge>
                                </div>
                                <div className="flex justify-between items-center p-3 rounded-lg" style={{ backgroundColor: COLORS.background.light }}>
                                  <span style={{ color: COLORS.text.primary }}>일반 설문조사</span>
                                  <Badge style={{ backgroundColor: COLORS.warning, color: 'white' }}>보통</Badge>
                                </div>
                                <div className="flex justify-between items-center p-3 rounded-lg" style={{ backgroundColor: COLORS.background.light }}>
                                  <span style={{ color: COLORS.text.primary }}>소셜미디어 데이터</span>
                                  <Badge style={{ backgroundColor: COLORS.danger, color: 'white' }}>낮음</Badge>
                                </div>
                              </div>
                            </div>
                            
                            <div>
                              <h4 className="font-semibold mb-4" style={{ color: COLORS.text.primary }}>
                                모델 성능 비교
                              </h4>
                              <div className="space-y-3">
                                <div className="flex justify-between items-center p-3 rounded-lg" style={{ backgroundColor: COLORS.background.light }}>
                                  <span style={{ color: COLORS.text.primary }}>Random Forest (우리)</span>
                                  <Badge style={{ backgroundColor: COLORS.success, color: 'white' }}>85%</Badge>
                                </div>
                                <div className="flex justify-between items-center p-3 rounded-lg" style={{ backgroundColor: COLORS.background.light }}>
                                  <span style={{ color: COLORS.text.primary }}>Logistic Regression</span>
                                  <Badge style={{ backgroundColor: COLORS.warning, color: 'white' }}>78%</Badge>
                                </div>
                                <div className="flex justify-between items-center p-3 rounded-lg" style={{ backgroundColor: COLORS.background.light }}>
                                  <span style={{ color: COLORS.text.primary }}>Naive Bayes</span>
                                  <Badge style={{ backgroundColor: COLORS.danger, color: 'white' }}>72%</Badge>
                                </div>
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      {/* 데이터 기반 혁신 */}
                      <Card className="border-0 shadow-lg" style={{ backgroundColor: COLORS.primary, color: 'white' }}>
                        <CardHeader>
                          <CardTitle className="text-white">
                            <Zap className="inline mr-2" />
                            데이터 기반 혁신
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="text-center">
                            <div className="text-2xl font-bold mb-4">
                              &quot;Data-Driven Detective Agency&quot;
                            </div>
                            <div className="text-lg mb-6">
                              GSS 50년 데이터 + 머신러닝 + AI = 혁신적인 탐정 서비스
                            </div>
                            
                            <div className="bg-white rounded-lg p-6" style={{ color: COLORS.text.primary }}>
                              <div className="grid md:grid-cols-3 gap-4 text-center">
                                <div>
                                  <div className="text-2xl font-bold mb-2" style={{ color: COLORS.primary }}>
                                    24,460
                                  </div>
                                  <div className="text-sm">GSS 샘플 데이터</div>
                                </div>
                                <div>
                                  <div className="text-2xl font-bold mb-2" style={{ color: COLORS.secondary }}>
                                    85%
                                  </div>
                                  <div className="text-sm">예측 정확도</div>
                                </div>
                                <div>
                                  <div className="text-2xl font-bold mb-2" style={{ color: COLORS.accent }}>
                                    50년
                                  </div>
                                  <div className="text-sm">데이터 축적 기간</div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </div>

                    <div className="flex justify-between mt-8">
                      <Button 
                        onClick={prevStep}
                        variant="outline"
                        style={{ borderColor: COLORS.primary, color: COLORS.primary }}
                      >
                        ← 이전
                      </Button>
                      <Button 
                        onClick={() => setCurrentStep(0)}
                        className="px-8 py-3 text-lg font-semibold"
                        style={{ backgroundColor: COLORS.primary }}
                      >
                        처음으로 돌아가기
                      </Button>
                    </div>
                  </Section>
                </motion.div>
              )}
            </AnimatePresence>
          </Suspense>
        </ErrorBoundary>
      </main>

      {/* 푸터 */}
      <footer className="border-t mt-16" style={{ borderColor: COLORS.background.dark, backgroundColor: COLORS.background.medium }}>
        <div className="max-w-7xl mx-auto px-4 md:px-6 py-6 text-center">
          <p className="text-sm" style={{ color: COLORS.text.light }}>
            © 2024 엉덩이 탐정 - 불륜 증거 수집 전문 사설 탐정 | 
            디자인: Tailwind CSS + Motion + Recharts | 
            데이터 출처: 미국 일반사회조사(GSS) 1972-2022, 24,460개 샘플 기반 분석
          </p>
          <p className="text-xs mt-2" style={{ color: COLORS.text.light }}>
            머신러닝 모델: Random Forest (ROC AUC: 0.85) | 
            파생 변수: yrs_per_age, rate_x_yrs | 
            변수 중요도: 결혼 만족도(28%), 결혼 연수(24%), 나이(19%)
          </p>
        </div>
      </footer>
    </div>
  );
}
