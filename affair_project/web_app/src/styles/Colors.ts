export const COLORS = {
  // 기본 브랜드 컬러
  primary: "#2563eb",      // blue-600 - 남성 데이터, 변수 중요도
  secondary: "#4f46e5",    // indigo-600
  violet: "#7c3aed",       // violet-600
  success: "#10b981",      // emerald-600 - 결혼 연수, 종교 활동
  warning: "#f59e0b",      // amber-600 - 결혼 만족도, 결혼만족도×결혼연수
  danger: "#ef4444",       // red-600 - 여성 데이터, 결혼연수/나이 비율
  teal: "#14b8a6",         // teal-500 - 자녀 수, 세대별 코호트
  rose: "#e11d48",         // rose-600 - 직업 등급

  // 차트 컬러 팔레트 (statistics/page.tsx 기반)
  chart: {
    blue: "#2563eb",       // blue-600 - 남성 데이터, 변수 중요도
    red: "#ef4444",        // red-500 - 여성 데이터, 결혼연수/나이 비율
    green: "#10b981",      // emerald-500 - 결혼 연수, 종교 활동
    amber: "#f59e0b",      // amber-500 - 결혼 만족도, 결혼만족도×결혼연수
    violet: "#8b5cf6",     // violet-500 - 결혼 만족도
    teal: "#14b8a6",       // teal-500 - 자녀 수, 세대별 코호트
    rose: "#e11d48",       // rose-600 - 직업 등급
  },

  // 다크 테마 배경 컬러 (statistics/page.tsx 기반)
  dark: {
    background: {
      primary: "#0f172a",    // slate-950 - 메인 배경 시작
      secondary: "#1e293b",  // slate-800 - 메인 배경 끝
      card: "rgba(30, 41, 59, 0.6)",  // slate-900/60 - 카드 배경
      cardLight: "rgba(30, 41, 59, 0.5)", // slate-900/50 - 히어로 차트 배경
      icon: "rgba(51, 65, 85, 0.8)",  // slate-800/80 - 아이콘 배경
      button: "#1e293b",    // slate-800 - 버튼 배경
      buttonHover: "#334155", // slate-700 - 버튼 호버 배경
    },
    border: "#1e293b",      // slate-800 - 테두리
    grid: "#1f2937",        // slate-800 - 차트 그리드
    axis: "#94a3b8",        // slate-400 - 차트 축
  },

  // 라이트 테마 배경 컬러 (기존 detective-story 기반)
  light: {
    background: {
      primary: "#eff6ff",    // blue-50
      secondary: "#eef2ff",  // indigo-50
      tertiary: "#dbeafe",   // blue-100
    },
  },

  // 텍스트 컬러 (statistics/page.tsx 기반)
  text: {
    // 다크 테마
    dark: {
      primary: "#ffffff",     // white - 제목, 주요 숫자
      secondary: "#f1f5f9",  // slate-100 - 헤더, 카드 제목
      tertiary: "#e2e8f0",   // slate-200 - 아이콘 컨테이너
      quaternary: "#cbd5e1", // slate-300 - 부제목, 설명 텍스트
      quinary: "#94a3b8",    // slate-400 - 출처, 라벨, 보조 텍스트
    },
    // 라이트 테마
    light: {
      primary: "#1e40af",    // blue-800
      secondary: "#3730a3",  // indigo-800
      tertiary: "#6366f1",   // indigo-500
      quaternary: "#4f46e5", // indigo-600 - 부제목, 설명 텍스트
      quinary: "#6366f1",    // indigo-500 - 출처, 라벨, 보조 텍스트
    },
  },

  // 강조 컬러 (statistics/page.tsx 기반)
  accent: {
    sky: "#0ea5e9",         // sky-500 - 주요 강조 (불륜 예측 모델, Shield 아이콘)
    green: "#22c55e",       // green-500 - 긍정적 요소 (TrendingUp 아이콘)
    purple: "#a855f7",      // purple-500 - 정보 요소 (BookOpen 아이콘)
  },

  // 차트 툴팁 컬러 (statistics/page.tsx 기반)
  tooltip: {
    background: "#0b1220",   // 매우 어두운 색
    border: "#1f2937",       // slate-800
    text: "#e2e8f0",         // slate-200
  },

  // 기존 컬러 (하위 호환성)
  white: "#ffffff",
};

// 차트용 컬러 배열 (statistics/page.tsx와 동일)
export const CHART_COLORS = [
  COLORS.chart.blue,    // 0: 남성 데이터, 변수 중요도
  COLORS.chart.red,     // 1: 여성 데이터, 결혼연수/나이 비율
  COLORS.chart.green,   // 2: 결혼 연수, 종교 활동
  COLORS.chart.amber,   // 3: 결혼 만족도, 결혼만족도×결혼연수
  COLORS.chart.violet,  // 4: 결혼 만족도
  COLORS.chart.teal,    // 5: 자녀 수, 세대별 코호트
  COLORS.chart.rose,    // 6: 직업 등급
];

// 사용 예시를 위한 타입 정의
export type ColorTheme = 'light' | 'dark';
export type ChartColorIndex = 0 | 1 | 2 | 3 | 4 | 5 | 6;
