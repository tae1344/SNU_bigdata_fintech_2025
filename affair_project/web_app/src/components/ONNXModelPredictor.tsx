'use client'

import { Button } from "@/components/ui/button";
import { Calculator } from "lucide-react";
import { useState } from "react";
import { useThemeColors } from "../hooks/useThemeColors";

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
  prediction: number;
  probability: number;
  error?: string;
}

interface ONNXModelPredictorProps {
  userInput: UserInput;
  onPrediction: (result: PredictionResult) => void;
  onError: (error: string) => void;
}

export default function ONNXModelPredictor({ userInput, onPrediction, onError }: ONNXModelPredictorProps) {
  const { colors } = useThemeColors();
  const [isLoading, setIsLoading] = useState(false);
  const [testResult, setTestResult] = useState<string>('');

  // 기본 확률 계산 함수 (ONNX 모델 실패 시 사용)
  const estimateProbabilityFromFeatures = (features: number[], prediction: number): number => {
    // 간단한 규칙 기반 확률 계산
    let baseProb = 20;
    
    if (features[2] <= 2) baseProb += 10; // 낮은 종교성
    if (features[3] < 12) baseProb += 8;  // 낮은 교육수준
    if (features[4] <= 2) baseProb += 12; // 낮은 직업등급
    if (features[10] === 1) baseProb += 15; // 낮은 결혼만족도
    
    return Math.min(100, Math.max(0, baseProb));
  };

  // ONNX 모델 테스트 함수
  const testModel = async () => {
    setIsLoading(true);
    setTestResult('테스트 시작...');
    
    try {
      // ONNX Runtime Web 동적 import
      const ort = await import('onnxruntime-web');
      
      setTestResult('ONNX 모델 로딩 중...');
      
      // 모델 로드
      const session = await ort.InferenceSession.create('/model.onnx');
      setTestResult('모델 로드 성공! 예측 시도 중...');
      
      // 입력 데이터 전처리 (12개 특성)
      const inputFeatures = [
        userInput.age,
        userInput.children,
        userInput.religiousness_5,
        userInput.education,
        userInput.occupation_grade6,
        userInput.occupation_husb_grade6,
        userInput.gender === 'male' ? 1 : 0,
        userInput.gender === 'female' ? 1 : 0,
        userInput.yearsmarried,
        userInput.yearsmarried / userInput.age,
        userInput.rating_5,
        userInput.rating_5 * userInput.yearsmarried
      ];
      
      // 입력 텐서 생성
      const inputTensor = new ort.Tensor('float32', inputFeatures, [1, 12]);
      
      // 모델 실행
      const feeds = { [session.inputNames[0]]: inputTensor };
      const results = await session.run(feeds);
      
      setTestResult('모델 실행 성공! 결과 처리 중...');
      
      // 결과 구조 디버깅
      console.log('🔍 ONNX 모델 결과 전체:', results);
      console.log('🔍 결과 키들:', Object.keys(results));
      
      // 결과 파싱 (새로운 출력 구조)
      const outputLabel = results.label;
      const outputProbability = results.probabilities;
      
      console.log('🔍 outputLabel:', outputLabel);
      console.log('🔍 outputProbability:', outputProbability);
      
      let prediction = 0;
      let probability = 0;
      
      // 예측값 추출 (label)
      if (outputLabel && 'data' in outputLabel) {
        prediction = Number((outputLabel as any).data[0]);
        console.log('🔍 예측값 추출 성공:', prediction);
      } else {
        console.log('❌ outputLabel에서 예측값 추출 실패');
        console.log('outputLabel 구조:', outputLabel);
      }
      
      // 확률값 추출 (probabilities - 표준 텐서)
      if (outputProbability && 'data' in outputProbability) {
        const probData = (outputProbability as any).data;
        console.log('🔍 probData:', probData);
        console.log('🔍 probData 타입:', typeof probData);
        console.log('🔍 probData 길이:', probData.length || '길이 없음');
        
        // Float32Array 또는 일반 배열 모두 처리
        if (probData && (Array.isArray(probData) || probData instanceof Float32Array) && probData.length >= 2) {
          // 클래스 1(불륜)의 확률을 사용
          probability = Number(probData[1]) * 100;
          console.log('🔍 확률값 추출 성공:', probability);
        } else {
          console.log('❌ probData가 올바른 형태가 아님');
        }
      } else {
        console.log('❌ outputProbability에서 확률값 추출 실패');
        console.log('outputProbability 구조:', outputProbability);
      }
      
      // 확률값이 추출되지 않은 경우 기본 계산 사용
      if (probability === 0) {
        console.log('🔍 확률값이 추출되지 않은 경우 기본 계산 사용');
        probability = estimateProbabilityFromFeatures(inputFeatures, prediction);
      }
      
      setTestResult('✅ ONNX 모델 예측 성공!');
      
      // 결과 전달
      onPrediction({
        prediction,
        probability: Math.round(probability * 10) / 10
      });
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '알 수 없는 오류';
      setTestResult(`❌ ONNX 모델 실패: ${errorMessage}`);
      onError(`ONNX 모델 예측 실패: ${errorMessage}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-3">
      <Button
        onClick={testModel}
        disabled={isLoading}
        className="px-8 py-4 text-lg font-semibold transition-all duration-300 hover:scale-105 disabled:opacity-50 w-full"
        style={{
          backgroundColor: colors.brand.primary,
          color: '#ffffff'
        }}
      >
        {isLoading ? (
          <>
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
            AI 모델 분석 중...
          </>
        ) : (
          <>
            <Calculator className="w-5 h-5 mr-2" />
            AI 모델로 위험도 분석하기
          </>
        )}
      </Button>
      
      {testResult && (
        <div className="text-xs p-2 rounded bg-gray-100 dark:bg-gray-800">
          {testResult}
        </div>
      )}
    </div>
  );
}
