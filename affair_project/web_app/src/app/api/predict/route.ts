// src/app/api/predict/route.ts
import { NextRequest, NextResponse } from 'next/server';

// 사용자 입력 타입
interface UserInput {
  age: number;
  yearsmarried: number;
  children: number;
  religiousness_5: number;
  education: number;
  occupation_grade6: number;
  occupation_husb_grade6: number;
  rating_5: number;
  gender: 'male' | 'female';
}

// 예측 결과 타입
interface PredictionResult {
  probability: number;
  riskLevel: 'low' | 'medium' | 'high';
  factors: string[];
  recommendations: string[];
  model_confidence: 'high' | 'medium' | 'low';
  fallback?: boolean; // 폴백 사용 여부
  fallback_reason?: string; // 폴백 사유
}

// 규칙 기반 예측 함수 (GSS 데이터 기반)
function calculatePredictionWithRules(userInput: UserInput): PredictionResult {
  let baseProbability = 17.77; // GSS 데이터의 전체 불륜률
  
  // 나이 요인 (GSS 데이터 기반)
  if (userInput.age < 30) baseProbability -= 5.5;
  else if (userInput.age >= 50) baseProbability += 8.0;
  
  // 결혼 연수 요인 (GSS 데이터 기반)
  if (userInput.yearsmarried < 5) baseProbability -= 3.6;
  else if (userInput.yearsmarried >= 20) baseProbability += 7.6;
  
  // 자녀 수 요인 (GSS 데이터 기반)
  if (userInput.children === 0) baseProbability += 3.1;
  else if (userInput.children >= 4) baseProbability -= 4.8;
  
  // 종교성 요인 (GSS 데이터 기반)
  if (userInput.religiousness_5 <= 2) baseProbability += 6.8;
  else if (userInput.religiousness_5 >= 4) baseProbability -= 4.5;
  
  // 교육 수준 요인 (GSS 데이터 기반)
  if (userInput.education < 12) baseProbability += 4.7;
  else if (userInput.education >= 16) baseProbability -= 2.9;
  
  // 직업 등급 요인 (GSS 데이터 기반)
  if (userInput.occupation_grade6 <= 2) baseProbability += 6.8;
  else if (userInput.occupation_grade6 >= 5) baseProbability -= 2.1;
  
  // 결혼 만족도 요인 (GSS 데이터 기반)
  if (userInput.rating_5 === 1) baseProbability += 16.4;
  else if (userInput.rating_5 === 3) baseProbability += 0.4;
  else if (userInput.rating_5 === 5) baseProbability -= 5.6;
  
  // 성별 요인 (GSS 데이터 기반)
  if (userInput.gender === 'male') baseProbability += 4.3;
  
  // 파생 변수 요인
  const yrsPerAge = userInput.yearsmarried / userInput.age;
  if (yrsPerAge > 0.7) baseProbability += 7.3;
  
  const rateXYrs = userInput.rating_5 * userInput.yearsmarried;
  if (rateXYrs < 50) baseProbability += 8.5;
  else if (rateXYrs > 150) baseProbability -= 4.1;
  
  // 확률 범위 제한
  const finalProbability = Math.max(0, Math.min(100, baseProbability));
  
  // 위험도 판정
  let riskLevel: 'low' | 'medium' | 'high';
  if (finalProbability < 20) riskLevel = 'low';
  else if (finalProbability < 35) riskLevel = 'medium';
  else riskLevel = 'high';
  
  // 주요 요인 분석
  const factors = analyzeFactors(userInput);
  
  // 권장사항
  const recommendations = generateRecommendations(userInput);
  
  return {
    probability: Math.round(finalProbability * 10) / 10,
    riskLevel,
    factors,
    recommendations,
    model_confidence: 'medium',  // 규칙 기반 예측
    fallback: true,
    fallback_reason: '서버 사이드에서 안정적인 규칙 기반 예측 사용'
  };
}

// 요인 분석
function analyzeFactors(userInput: UserInput): string[] {
  const factors: string[] = [];
  
  if (userInput.rating_5 === 1) factors.push('낮은 결혼 만족도');
  if (userInput.yearsmarried >= 20) factors.push('긴 결혼 기간');
  if (userInput.religiousness_5 <= 2) factors.push('낮은 종교성');
  if (userInput.occupation_grade6 <= 2) factors.push('낮은 직업 등급');
  if (userInput.children === 0) factors.push('자녀 없음');
  
  const yrsPerAge = userInput.yearsmarried / userInput.age;
  if (yrsPerAge > 0.7) factors.push('일찍 결혼');
  
  return factors;
}

// 권장사항 생성
function generateRecommendations(userInput: UserInput): string[] {
  const recommendations: string[] = [];
  
  if (userInput.rating_5 === 1) recommendations.push('결혼 상담 프로그램 참여');
  if (userInput.religiousness_5 <= 2) recommendations.push('종교 활동 참여 고려');
  if (userInput.children === 0) recommendations.push('가족 관계 강화');
  if (userInput.occupation_grade6 <= 2) recommendations.push('직업 개발 프로그램');
  
  return recommendations;
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { userInput } = body;

    console.log('Received user input:', userInput);
    
    // 입력 검증
    if (!userInput || typeof userInput !== 'object') {
      return NextResponse.json(
        { error: 'Invalid user input' },
        { status: 400 }
      );
    }

    // 필수 필드 확인
    const requiredFields = ['age', 'yearsmarried', 'children', 'religiousness_5', 'education', 'occupation_grade6', 'rating_5', 'gender'];
    for (const field of requiredFields) {
      if (userInput[field] === undefined || userInput[field] === null) {
        return NextResponse.json(
          { error: `Missing required field: ${field}` },
          { status: 400 }
        );
      }
    }

    // 규칙 기반 예측 수행 (안정적이고 빠름)
    console.log('규칙 기반 예측 수행 중...');
    const prediction = calculatePredictionWithRules(userInput);
    
    console.log('규칙 기반 예측 결과:', prediction);
    
    return NextResponse.json(prediction);

  } catch (error) {
    console.error('API error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}