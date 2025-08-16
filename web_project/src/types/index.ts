export type Filters = {
  rate_marriage: 1 | 2 | 3 | 4 | 5; // 결혼 만족도
  age: number; // 나이
  yrs_married: number; // 결혼 기간
  children: number; // 자녀 수
  religious: 1 | 2 | 3 | 4; // 종교 성향
  educ: number; // 교육 수준 코드
  occupation: number; // 직업 코드
  occupation_husb: number; // 배우자 직업 코드
};

export type ScoreResult = {
  score: number; // 0-100
  probability: number; // 0-1
  contributions: { key: keyof Filters; value: number }[];
};

export type AppStep = "landing" | "filters" | "result" | "compare";

export type AppState = {
  currentStep: AppStep;
  filters: Filters;
  result: ScoreResult | null;
  previousResult: ScoreResult | null; // 비교용
  loading: boolean;
};

export type AppActions = {
  setCurrentStep: (step: AppStep) => void;
  setLoading: (loading: boolean) => void;
  updateFilter: <K extends keyof Filters>(key: K, value: Filters[K]) => void;
  setResult: (result: ScoreResult) => void;
  setPreviousResult: (result: ScoreResult) => void;
  resetToLanding: () => void;
};
