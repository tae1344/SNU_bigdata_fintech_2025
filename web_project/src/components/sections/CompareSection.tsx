"use client";
import React from "react";
import { Button } from "@/components/ui/button";
import { useStore } from "@/store/useStore";
import { GaugeChart } from "@/components/legacy/GaugeChart";
import { AvatarScene } from "@/components/legacy/AvatarScene";
import { getScoreDescription, getScoreColor } from "@/lib/scoring";
import { EthicsNotice } from "@/components/legacy/EthicsNotice";

export function CompareSection() {
  const { result, previousResult, filters, setCurrentStep } = useStore();

  if (!result || !previousResult) return null;

  const handleNewTest = () => {
    setCurrentStep("landing");
  };

  const scoreDifference = result.score - previousResult.score;
  const scoreChangeColor = scoreDifference > 0 ? "text-red-600" : "text-green-600";
  const scoreChangeIcon = scoreDifference > 0 ? "↗️" : "↘️";

  return (
    <div className="flex flex-col items-center justify-center min-h-screen px-4 py-8">
      <div className="w-full max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">결과 비교</h1>
          <p className="text-xl text-gray-600">
            두 시나리오의 결과를 비교해보세요
          </p>
        </div>

        {/* Comparison Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Previous Result */}
          <div className="bg-gray-50 rounded-lg p-6">
            <h3 className="text-2xl font-semibold text-center mb-6 text-gray-600">
              이전 결과
            </h3>
            
            <div className="text-center mb-6">
              <div className={`text-5xl font-bold mb-2 ${getScoreColor(previousResult.score)}`}>
                {previousResult.score}
              </div>
              <div className="text-xl text-gray-700 mb-2">
                {getScoreDescription(previousResult.score)}
              </div>
            </div>

            <div className="flex justify-center mb-6">
              <GaugeChart value={previousResult.score} label="Previous" />
            </div>

            <div className="h-32">
              <AvatarScene probability={previousResult.probability} />
            </div>
          </div>

          {/* Current Result */}
          <div className="bg-blue-50 rounded-lg p-6 border-2 border-blue-200">
            <h3 className="text-2xl font-semibold text-center mb-6 text-blue-800">
              현재 결과
            </h3>
            
            <div className="text-center mb-6">
              <div className={`text-5xl font-bold mb-2 ${getScoreColor(result.score)}`}>
                {result.score}
              </div>
              <div className="text-xl text-gray-700 mb-2">
                {getScoreDescription(result.score)}
              </div>
            </div>

            <div className="flex justify-center mb-6">
              <GaugeChart value={result.score} label="Current" />
            </div>

            <div className="h-32">
              <AvatarScene probability={result.probability} />
            </div>
          </div>
        </div>

        {/* Score Change Analysis */}
        <div className="bg-white rounded-lg border p-6 mb-8">
          <h3 className="text-2xl font-semibold text-center mb-6">변화 분석</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-center">
            <div>
              <div className="text-3xl font-bold text-gray-700 mb-2">
                {scoreChangeIcon} {Math.abs(scoreDifference)}
              </div>
              <div className="text-sm text-gray-600">점수 변화</div>
            </div>
            
            <div>
              <div className={`text-3xl font-bold mb-2 ${scoreChangeColor}`}>
                {scoreDifference > 0 ? "증가" : "감소"}
              </div>
              <div className="text-sm text-gray-600">변화 방향</div>
            </div>
            
            <div>
              <div className="text-3xl font-bold text-gray-700 mb-2">
                {Math.abs(scoreDifference) > 20 ? "큰 변화" : 
                 Math.abs(scoreDifference) > 10 ? "중간 변화" : "작은 변화"}
              </div>
              <div className="text-sm text-gray-600">변화 크기</div>
            </div>
          </div>

          {/* Change Description */}
          <div className="mt-6 text-center">
            <p className="text-lg text-gray-700">
              {scoreDifference > 0 
                ? `점수가 ${scoreDifference}점 증가했습니다. 조건 변화가 확률에 긍정적 영향을 미쳤습니다.`
                : `점수가 ${Math.abs(scoreDifference)}점 감소했습니다. 조건 변화가 확률에 부정적 영향을 미쳤습니다.`
              }
            </p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-8">
          <Button onClick={handleNewTest} size="lg">
            새로 테스트하기
          </Button>
        </div>

        <EthicsNotice className="mt-8" />
      </div>
    </div>
  );
}
