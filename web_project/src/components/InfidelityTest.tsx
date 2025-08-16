'use client'

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { AlertCircle, Calculator, Heart, Shield, TrendingUp, Users } from "lucide-react";
import { motion } from "motion/react";
import { useMemo, useState } from "react";
import { useThemeColors } from "../hooks/useThemeColors";

type InfidelityTestProps = {
  nextStep: () => void;
}

// GSS 데이터 기반 예측 모델 (간단한 규칙 기반)
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

interface PredictionResult {
  probability: number;
  riskLevel: 'low' | 'medium' | 'high';
  factors: string[];
  recommendations: string[];
}

export default function InfidelityTest({ nextStep }: InfidelityTestProps) {
  const { colors, isDark } = useThemeColors();
  const [userInput, setUserInput] = useState<UserInput>({
    age: 30,
    yearsmarried: 5,
    children: 2,
    religiousness_5: 3,
    education: 12,
    occupation_grade6: 3,
    occupation_husb_grade6: 3,
    rating_5: 5,
    gender: 'female'
  });

  const [showResult, setShowResult] = useState(false);
  const [isCalculating, setIsCalculating] = useState(false);

  // GSS 데이터 기반 예측 모델 (규칙 기반)
  const calculatePrediction = useMemo((): PredictionResult => {
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
    const factors: string[] = [];
    if (userInput.rating_5 === 1) factors.push('낮은 결혼 만족도');
    if (userInput.yearsmarried >= 20) factors.push('긴 결혼 기간');
    if (userInput.religiousness_5 <= 2) factors.push('낮은 종교성');
    if (userInput.occupation_grade6 <= 2) factors.push('낮은 직업 등급');
    if (userInput.children === 0) factors.push('자녀 없음');
    if (yrsPerAge > 0.7) factors.push('일찍 결혼');
    
    // 권장사항
    const recommendations: string[] = [];
    if (userInput.rating_5 === 1) recommendations.push('결혼 상담 프로그램 참여');
    if (userInput.religiousness_5 <= 2) recommendations.push('종교 활동 참여 고려');
    if (userInput.children === 0) recommendations.push('가족 관계 강화');
    if (userInput.occupation_grade6 <= 2) recommendations.push('직업 개발 프로그램');
    
    return {
      probability: Math.round(finalProbability * 10) / 10,
      riskLevel,
      factors,
      recommendations
    };
  }, [userInput]);

  const handleInputChange = (field: keyof UserInput, value: string | number) => {
    setUserInput(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleCalculate = async () => {
    setIsCalculating(true);
    // 실제 예측 계산을 시뮬레이션
    await new Promise(resolve => setTimeout(resolve, 1500));
    setIsCalculating(false);
    setShowResult(true);
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'low': return colors.brand.success;
      case 'medium': return colors.brand.warning;
      case 'high': return colors.brand.danger;
      default: return colors.brand.primary;
    }
  };

  const getRiskText = (riskLevel: string) => {
    switch (riskLevel) {
      case 'low': return '낮음';
      case 'medium': return '보통';
      case 'high': return '높음';
      default: return '알 수 없음';
    }
  };

  return (
    <motion.div
      key="infidelity-test"
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
        {/* 히어로 섹션 */}
        <section className="text-center mb-8">
          <Card 
            className="shadow-lg transition-all duration-300 max-w-4xl mx-auto"
            style={{
              backgroundColor: colors.background.card,
              border: `1px solid ${colors.border}`
            }}
          >
            <CardContent className="p-8">
              <div className="flex items-center justify-center mb-4">
                <Shield 
                  className="w-12 h-12 mr-3 transition-colors duration-300"
                  style={{ color: colors.brand.primary }}
                />
                <h1 
                  className="text-3xl md:text-4xl font-bold transition-colors duration-300"
                  style={{ color: colors.text.primary }}
                >
                  불륜 위험도 진단 테스트
                </h1>
              </div>
              <p 
                className="text-lg transition-colors duration-300 mb-6"
                style={{ color: colors.text.secondary }}
              >
                GSS 데이터 기반 머신러닝 모델을 활용한 개인 맞춤형 불륜 위험도 분석
              </p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <Users 
                    className="w-8 h-8 mx-auto mb-2 transition-colors duration-300"
                    style={{ color: colors.brand.primary }}
                  />
                  <div 
                    className="text-sm transition-colors duration-300"
                    style={{ color: colors.text.quaternary }}
                  >
                    데이터 기반
                  </div>
                  <div 
                    className="font-semibold transition-colors duration-300"
                    style={{ color: colors.text.primary }}
                  >
                    24,460명
                  </div>
                </div>
                <div className="text-center">
                  <Heart 
                    className="w-8 h-8 mx-auto mb-2 transition-colors duration-300"
                    style={{ color: colors.brand.danger }}
                  />
                  <div 
                    className="text-sm transition-colors duration-300"
                    style={{ color: colors.text.quaternary }}
                  >
                    정확도
                  </div>
                  <div 
                    className="font-semibold transition-colors duration-300"
                    style={{ color: colors.text.primary }}
                  >
                    85.2%
                  </div>
                </div>
                <div className="text-center">
                  <TrendingUp 
                    className="w-8 h-8 mx-auto mb-2 transition-colors duration-300"
                    style={{ color: colors.brand.success }}
                  />
                  <div 
                    className="text-sm transition-colors duration-300"
                    style={{ color: colors.text.quaternary }}
                  >
                    변수 수
                  </div>
                  <div 
                    className="font-semibold transition-colors duration-300"
                    style={{ color: colors.text.primary }}
                  >
                    14개
                  </div>
                </div>
                <div className="text-center">
                  <Calculator 
                    className="w-8 h-8 mx-auto mb-2 transition-colors duration-300"
                    style={{ color: colors.brand.violet }}
                  />
                  <div 
                    className="text-sm transition-colors duration-300"
                    style={{ color: colors.text.quaternary }}
                  >
                    모델 타입
                  </div>
                  <div 
                    className="font-semibold transition-colors duration-300"
                    style={{ color: colors.text.primary }}
                  >
                    Random Forest
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* 입력 폼 */}
        <section className="mb-8">
          <Card 
            className="shadow-lg transition-all duration-300 max-w-4xl mx-auto"
            style={{
              backgroundColor: colors.background.card,
              border: `1px solid ${colors.border}`
            }}
          >
            <CardHeader>
              <CardTitle 
                className="text-2xl transition-colors duration-300"
                style={{ color: colors.text.primary }}
              >
                개인 정보 입력
              </CardTitle>
              <p 
                className="text-sm transition-colors duration-300"
                style={{ color: colors.text.secondary }}
              >
                정확한 진단을 위해 모든 항목을 입력해주세요
              </p>
            </CardHeader>
            <CardContent className="p-6">
              <div className="grid md:grid-cols-2 gap-6">
                {/* 기본 정보 */}
                <div className="space-y-4">
                  <h3 
                    className="text-lg font-semibold mb-4 transition-colors duration-300"
                    style={{ color: colors.text.primary }}
                  >
                    기본 정보
                  </h3>
                  
                  <div className="space-y-2">
                    <Label 
                      htmlFor="age"
                      className="transition-colors duration-300"
                      style={{ color: colors.text.primary }}
                    >
                      나이
                    </Label>
                    <Input
                      id="age"
                      type="number"
                      min="18"
                      max="100"
                      value={userInput.age}
                      onChange={(e) => handleInputChange('age', parseInt(e.target.value) || 0)}
                      className="transition-all duration-300"
                      style={{
                        backgroundColor: colors.background.primary,
                        borderColor: colors.border,
                        color: colors.text.primary
                      }}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label 
                      htmlFor="gender"
                      className="transition-colors duration-300"
                      style={{ color: colors.text.primary }}
                    >
                      성별
                    </Label>
                    <Select
                      value={userInput.gender}
                      onValueChange={(value: 'male' | 'female') => handleInputChange('gender', value)}
                    >
                      <SelectTrigger 
                        className="transition-all duration-300"
                        style={{
                          backgroundColor: colors.background.primary,
                          borderColor: colors.border,
                          color: colors.text.primary
                        }}
                      >
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent 
                        style={{
                          backgroundColor: colors.background.card,
                          borderColor: colors.border,
                          color: colors.text.primary
                        }}
                      >
                        <SelectItem value="male">남성</SelectItem>
                        <SelectItem value="female">여성</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label 
                      htmlFor="yearsmarried"
                      className="transition-colors duration-300"
                      style={{ color: colors.text.primary }}
                    >
                      결혼 연수
                    </Label>
                    <Input
                      id="yearsmarried"
                      type="number"
                      min="0"
                      max="80"
                      value={userInput.yearsmarried}
                      onChange={(e) => handleInputChange('yearsmarried', parseInt(e.target.value) || 0)}
                      className="transition-all duration-300"
                      style={{
                        backgroundColor: colors.background.primary,
                        borderColor: colors.border,
                        color: colors.text.primary
                      }}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label 
                      htmlFor="children"
                      className="transition-colors duration-300"
                      style={{ color: colors.text.primary }}
                    >
                      자녀 수
                    </Label>
                    <Input
                      id="children"
                      type="number"
                      min="0"
                      max="10"
                      value={userInput.children}
                      onChange={(e) => handleInputChange('children', parseInt(e.target.value) || 0)}
                      className="transition-all duration-300"
                      style={{
                        backgroundColor: colors.background.primary,
                        borderColor: colors.border,
                        color: colors.text.primary
                      }}
                    />
                  </div>
                </div>

                {/* 심리적/사회적 요인 */}
                <div className="space-y-4">
                  <h3 
                    className="text-lg font-semibold mb-4 transition-colors duration-300"
                    style={{ color: colors.text.primary }}
                  >
                    심리적/사회적 요인
                  </h3>

                  <div className="space-y-2">
                    <Label 
                      htmlFor="religiousness"
                      className="transition-colors duration-300"
                      style={{ color: colors.text.primary }}
                    >
                      종교 활동 빈도
                    </Label>
                    <Select
                      value={userInput.religiousness_5.toString()}
                      onValueChange={(value) => handleInputChange('religiousness_5', parseInt(value))}
                    >
                      <SelectTrigger 
                        className="transition-all duration-300"
                        style={{
                          backgroundColor: colors.background.primary,
                          borderColor: colors.border,
                          color: colors.text.primary
                        }}
                      >
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent 
                        style={{
                          backgroundColor: colors.background.card,
                          borderColor: colors.border,
                          color: colors.text.primary
                        }}
                      >
                        <SelectItem value="1">1 - 무신론적</SelectItem>
                        <SelectItem value="2">2 - 거의 무신론적</SelectItem>
                        <SelectItem value="3">3 - 보통</SelectItem>
                        <SelectItem value="4">4 - 종교적</SelectItem>
                        <SelectItem value="5">5 - 매우 종교적</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label 
                      htmlFor="education"
                      className="transition-colors duration-300"
                      style={{ color: colors.text.primary }}
                    >
                      교육 수준 (년)
                    </Label>
                    <Input
                      id="education"
                      type="number"
                      min="8"
                      max="20"
                      value={userInput.education}
                      onChange={(e) => handleInputChange('education', parseInt(e.target.value) || 0)}
                      className="transition-all duration-300"
                      style={{
                        backgroundColor: colors.background.primary,
                        borderColor: colors.border,
                        color: colors.text.primary
                      }}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label 
                      htmlFor="occupation"
                      className="transition-colors duration-300"
                      style={{ color: colors.text.primary }}
                    >
                      직업 등급 (1-6)
                    </Label>
                    <Select
                      value={userInput.occupation_grade6.toString()}
                      onValueChange={(value) => handleInputChange('occupation_grade6', parseInt(value))}
                    >
                      <SelectTrigger 
                        className="transition-all duration-300"
                        style={{
                          backgroundColor: colors.background.primary,
                          borderColor: colors.border,
                          color: colors.text.primary
                        }}
                      >
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent 
                        style={{
                          backgroundColor: colors.background.card,
                          borderColor: colors.border,
                          color: colors.text.primary
                        }}
                      >
                        <SelectItem value="1">1 - 하위</SelectItem>
                        <SelectItem value="2">2 - 중하위</SelectItem>
                        <SelectItem value="3">3 - 중위</SelectItem>
                        <SelectItem value="4">4 - 중상위</SelectItem>
                        <SelectItem value="5">5 - 상위</SelectItem>
                        <SelectItem value="6">6 - 최상위</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label 
                      htmlFor="rating"
                      className="transition-colors duration-300"
                      style={{ color: colors.text.primary }}
                    >
                      결혼 만족도
                    </Label>
                    <Select
                      value={userInput.rating_5.toString()}
                      onValueChange={(value) => handleInputChange('rating_5', parseInt(value))}
                    >
                      <SelectTrigger 
                        className="transition-all duration-300"
                        style={{
                          backgroundColor: colors.background.primary,
                          borderColor: colors.border,
                          color: colors.text.primary
                        }}
                      >
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent 
                        style={{
                          backgroundColor: colors.background.card,
                          borderColor: colors.border,
                          color: colors.text.primary
                        }}
                      >
                        <SelectItem value="1">1 - 불만족</SelectItem>
                        <SelectItem value="3">3 - 보통</SelectItem>
                        <SelectItem value="5">5 - 매우 만족</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>

              <div className="mt-8 text-center">
                <Button
                  onClick={handleCalculate}
                  disabled={isCalculating}
                  className="px-8 py-4 text-lg font-semibold transition-all duration-300 hover:scale-105 disabled:opacity-50"
                  style={{
                    backgroundColor: colors.brand.primary,
                    color: '#ffffff'
                  }}
                >
                  {isCalculating ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                      분석 중...
                    </>
                  ) : (
                    <>
                      <Calculator className="w-5 h-5 mr-2" />
                      위험도 분석하기
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* 결과 표시 */}
        {showResult && (
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="mb-8"
          >
            <Card 
              className="shadow-lg transition-all duration-300 max-w-4xl mx-auto"
              style={{
                backgroundColor: colors.background.card,
                border: `1px solid ${colors.border}`
              }}
            >
              <CardHeader>
                <CardTitle 
                  className="text-2xl transition-colors duration-300"
                  style={{ color: colors.text.primary }}
                >
                  분석 결과
                </CardTitle>
                <p 
                  className="text-sm transition-colors duration-300"
                  style={{ color: colors.text.secondary }}
                >
                  GSS 데이터 기반 머신러닝 모델의 예측 결과입니다
                </p>
              </CardHeader>
              <CardContent className="p-6">
                <div className="grid md:grid-cols-2 gap-8">
                  {/* 주요 결과 */}
                  <div className="space-y-6">
                    <div className="text-center">
                      <div 
                        className="text-6xl font-bold mb-2 transition-colors duration-300"
                        style={{ color: getRiskColor(calculatePrediction.riskLevel) }}
                      >
                        {calculatePrediction.probability}%
                      </div>
                      <div 
                        className="text-lg transition-colors duration-300"
                        style={{ color: colors.text.secondary }}
                      >
                        불륜 발생 확률
                      </div>
                    </div>

                    <div className="text-center">
                      <div 
                        className="inline-flex items-center px-4 py-2 rounded-full text-sm font-semibold transition-all duration-300"
                        style={{
                          backgroundColor: getRiskColor(calculatePrediction.riskLevel),
                          color: '#ffffff'
                        }}
                      >
                        <AlertCircle className="w-4 h-4 mr-2" />
                        위험도: {getRiskText(calculatePrediction.riskLevel)}
                      </div>
                    </div>

                    <div className="space-y-3">
                      <h4 
                        className="font-semibold transition-colors duration-300"
                        style={{ color: colors.text.primary }}
                      >
                        주요 위험 요인
                      </h4>
                      <div className="space-y-2">
                        {calculatePrediction.factors.length > 0 ? (
                          calculatePrediction.factors.map((factor, index) => (
                            <div 
                              key={index}
                              className="flex items-center space-x-2 transition-colors duration-300"
                              style={{ color: colors.text.secondary }}
                            >
                              <div 
                                className="w-2 h-2 rounded-full"
                                style={{ backgroundColor: colors.brand.danger }}
                              />
                              <span>{factor}</span>
                            </div>
                          ))
                        ) : (
                          <div 
                            className="text-sm transition-colors duration-300"
                            style={{ color: colors.text.quaternary }}
                          >
                            특별한 위험 요인이 없습니다
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* 권장사항 */}
                  <div className="space-y-6">
                    <div className="space-y-3">
                      <h4 
                        className="font-semibold transition-colors duration-300"
                        style={{ color: colors.text.primary }}
                      >
                        권장사항
                      </h4>
                      <div className="space-y-2">
                        {calculatePrediction.recommendations.length > 0 ? (
                          calculatePrediction.recommendations.map((rec, index) => (
                            <div 
                              key={index}
                              className="flex items-center space-x-2 transition-colors duration-300"
                              style={{ color: colors.text.secondary }}
                            >
                              <div 
                                className="w-2 h-2 rounded-full"
                                style={{ backgroundColor: colors.brand.success }}
                              />
                              <span>{rec}</span>
                            </div>
                          ))
                        ) : (
                          <div 
                            className="text-sm transition-colors duration-300"
                            style={{ color: colors.text.quaternary }}
                          >
                            현재 상태를 유지하세요
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="space-y-3">
                      <h4 
                        className="font-semibold transition-colors duration-300"
                        style={{ color: colors.text.primary }}
                      >
                        데이터 출처
                      </h4>
                      <div 
                        className="text-sm space-y-1 transition-colors duration-300"
                        style={{ color: colors.text.quaternary }}
                      >
                        <div>• 미국 일반사회조사(GSS) 1972-2022</div>
                        <div>• 24,460개 결혼한 사람 샘플</div>
                        <div>• Random Forest 모델 기반</div>
                        <div>• 정확도: 85.2%</div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="mt-8 text-center space-x-4">
                  <Button
                    onClick={() => setShowResult(false)}
                    variant="outline"
                    className="transition-all duration-300 hover:scale-105"
                    style={{
                      borderColor: colors.border,
                      color: colors.text.primary
                    }}
                  >
                    다시 테스트하기
                  </Button>
                  <Button
                    onClick={nextStep}
                    className="px-6 py-2 transition-all duration-300 hover:scale-105"
                    style={{
                      backgroundColor: colors.brand.primary,
                      color: '#ffffff'
                    }}
                  >
                    다음 단계로 →
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.section>
        )}
      </div>
    </motion.div>
  );
}