"use client";
import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { WorldInfidelityMap } from "@/components/legacy/WorldInfidelityMap";
import { InfidelityMap } from "@/components/legacy/InfidelityMap";
import { StatisticsDashboard } from "@/components/legacy/StatisticsDashboard";

export function MapsSection() {
  return (
    <section className="py-16 bg-white">
      <div className="container mx-auto px-4">
        {/* 섹션 헤더 */}
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            전 세계 바람지수 현황 & 데이터 분석
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            국가별, 지역별 바람지수를 인터랙티브 지도로 확인하고, 
            실제 데이터셋을 기반으로 한 통계 분석을 살펴보세요.
          </p>
          <div className="flex items-center justify-center gap-4 mt-6">
            <Badge variant="outline" className="text-sm">
              🌍 41개 국가 데이터
            </Badge>
            <Badge variant="outline" className="text-sm">
              🇺🇸 50개 주 데이터
            </Badge>
            <Badge variant="outline" className="text-sm">
              📊 6,368개 샘플 데이터
            </Badge>
            <Badge variant="outline" className="text-sm">
              📈 실시간 통계 분석
            </Badge>
          </div>
        </div>

        {/* 통계 대시보드 */}
        <div className="mb-16">
          <StatisticsDashboard />
        </div>

        {/* 지도 그리드 */}
        <div className="flex flex-col gap-8 mb-8">
          {/* 세계 지도 */}
          <WorldInfidelityMap />

          {/* 미국 지도 */}
          <InfidelityMap />
        </div>

        {/* 데이터 설명 */}
        <Card className="bg-gray-50 border-0">
          <CardContent className="pt-6">
            <div className="text-center space-y-4">
              <h3 className="text-xl font-semibold text-gray-900">
                📊 데이터 출처 및 주의사항
              </h3>
              <div className="text-sm text-gray-600 max-w-4xl mx-auto space-y-2">
                <p>
                  • <strong>세계 지도</strong>: 41개 국가의 자기보고형 설문 데이터 (2025년 기준)
                </p>
                <p>
                  • <strong>미국 지도</strong>: 50개 주의 통계 데이터 (자기보고형 설문 기반)
                </p>
                <p>
                  • <strong>통계 대시보드</strong>: 6,368개 샘플의 상세 분석 결과
                </p>
                <p>
                  • <strong>주의사항</strong>: 이 데이터는 엔터테인먼트 목적으로 제작되었으며, 
                  실제 통계나 연구 결과를 반영하지 않습니다.
                </p>
                <p>
                  • <strong>색상 범례</strong>: 파란색이 진할수록 높은 수치를 나타냅니다.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </section>
  );
}
