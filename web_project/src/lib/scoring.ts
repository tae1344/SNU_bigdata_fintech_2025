import { Filters, ScoreResult } from "@/types";

// 로지스틱 회귀 계수 (가상의 값, 실제로는 데이터셋에서 추출)
const COEFFICIENTS = {
  rate_marriage: -0.3, // 결혼 만족도가 높을수록 확률 감소
  age: -0.02, // 나이가 높을수록 확률 감소
  yrs_married: 0.05, // 결혼 기간이 길수록 확률 증가
  children: -0.1, // 자녀가 많을수록 확률 감소
  religious: -0.2, // 종교적일수록 확률 감소
  educ: -0.15, // 교육 수준이 높을수록 확률 감소
  occupation: 0.05, // 직업 코드에 따른 영향
  occupation_husb: 0.03, // 배우자 직업에 따른 영향
};

const INTERCEPT = -1.5; // 절편

// 각 변수의 평균값 (가상의 값)
const MEANS = {
  rate_marriage: 3,
  age: 35,
  yrs_married: 8,
  children: 1.5,
  religious: 2.5,
  educ: 3,
  occupation: 2.5,
  occupation_husb: 2.5,
};

export function calculateScore(filters: Filters): ScoreResult {
  // 로지스틱 회귀 계산
  let linearPredictor = INTERCEPT;
  
  Object.entries(filters).forEach(([key, value]) => {
    const coefficient = COEFFICIENTS[key as keyof Filters];
    linearPredictor += coefficient * value;
  });
  
  // 확률 계산 (로지스틱 함수)
  const probability = 1 / (1 + Math.exp(-linearPredictor));
  
  // 점수 변환 (0-100)
  const score = Math.round(probability * 100);
  
  // 영향 요인 계산
  const contributions = Object.entries(filters).map(([key, value]) => {
    const coefficient = COEFFICIENTS[key as keyof Filters];
    const mean = MEANS[key as keyof Filters];
    const contribution = coefficient * (value - mean);
    return { key: key as keyof Filters, value: contribution };
  });
  
  // 절대값 기준으로 정렬하여 상위 3개 선택
  const topContributions = contributions
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 3);
  
  return {
    score,
    probability,
    contributions: topContributions,
  };
}

export function getScoreDescription(score: number): string {
  if (score < 20) return "매우 낮음";
  if (score < 40) return "낮음";
  if (score < 60) return "보통";
  if (score < 80) return "높음";
  return "매우 높음";
}

export function getScoreColor(score: number): string {
  if (score < 20) return "text-green-600";
  if (score < 40) return "text-blue-600";
  if (score < 60) return "text-yellow-600";
  if (score < 80) return "text-orange-600";
  return "text-red-600";
}
