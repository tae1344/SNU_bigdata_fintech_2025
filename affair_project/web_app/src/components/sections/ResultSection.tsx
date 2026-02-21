"use client";
import { AvatarScene } from "@/components/legacy/AvatarScene";
import { ContributionsBar } from "@/components/legacy/ContributionsBar";
import { EthicsNotice } from "@/components/legacy/EthicsNotice";
import { GaugeChart } from "@/components/legacy/GaugeChart";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { getScoreColor } from "@/lib/scoring";
import { useStore } from "@/store/useStore";

export function ResultSection() {
  const { result, filters, setPreviousResult, setCurrentStep } = useStore();

  if (!result) return null;

  const handleCompare = () => {
    setPreviousResult(result);
    setCurrentStep("compare");
  };

  const handleNewTest = () => {
    setCurrentStep("landing");
  };

  const getScoreLevel = (score: number) => {
    if (score < 20) return { level: "매우 낮음", color: "bg-green-100 text-green-800" };
    if (score < 40) return { level: "낮음", color: "bg-blue-100 text-blue-800" };
    if (score < 60) return { level: "보통", color: "bg-yellow-100 text-yellow-800" };
    if (score < 80) return { level: "높음", color: "bg-orange-100 text-orange-800" };
    return { level: "매우 높음", color: "bg-red-100 text-red-800" };
  };

  const scoreLevel = getScoreLevel(result.score);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen px-4 py-8">
      <div className="w-full max-w-7xl mx-auto space-y-8">
        {/* Header Card */}
        <Card className="text-center">
          <CardHeader>
            <CardTitle className="text-4xl font-bold text-gray-900 mb-2">
              분석 결과
            </CardTitle>
            <p className="text-xl text-gray-600">
              입력하신 정보를 바탕으로 계산된 결과입니다
            </p>
          </CardHeader>
        </Card>

        {/* Main Result Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Score & Gauge */}
          <Card>
            <CardHeader>
              <CardTitle className="text-center text-2xl">점수 분석</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Score Display */}
              <div className="text-center space-y-4">
                <div className={`text-7xl font-bold ${getScoreColor(result.score)}`}>
                  {result.score}
                </div>
                <Badge className={`text-lg px-4 py-2 ${scoreLevel.color}`}>
                  {scoreLevel.level}
                </Badge>
                <div className="text-sm text-gray-500">
                  확률: {(result.probability * 100).toFixed(1)}%
                </div>
              </div>

              {/* Gauge Chart */}
              <div className="flex justify-center">
                <GaugeChart value={result.score} label="Score" />
              </div>

              {/* Score Progress Bar */}
              <div className="space-y-2">
                <div className="flex justify-between text-sm text-gray-600">
                  <span>낮음</span>
                  <span>보통</span>
                  <span>높음</span>
                </div>
                <Progress value={result.score} className="h-3" />
              </div>
            </CardContent>
          </Card>

          {/* Right Column - Avatar Animation */}
          <Card>
            <CardHeader>
              <CardTitle className="text-center text-2xl">캐릭터 반응</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <AvatarScene probability={result.probability} />
              
              {/* Animation Info */}
              <div className="text-center space-y-2">
                <Badge variant="outline" className="text-sm">
                  애니메이션 정보
                </Badge>
                <p className="text-sm text-gray-600">
                  확률이 높을수록 캐릭터가 더 멀리, 더 빠르게 도망갑니다
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Contributions Section */}
        <Card>
          <CardHeader>
            <CardTitle className="text-center text-2xl">주요 영향 요인</CardTitle>
            <p className="text-center text-gray-600">
              점수에 가장 큰 영향을 미친 요인들을 분석했습니다
            </p>
          </CardHeader>
          <CardContent>
            <ContributionsBar contributions={result.contributions} />
          </CardContent>
        </Card>

        {/* Input Summary Section */}
        <Card>
          <CardHeader>
            <CardTitle className="text-2xl">입력 정보 요약</CardTitle>
            <p className="text-gray-600">
              분석에 사용된 입력 값들을 확인하세요
            </p>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div className="space-y-2">
                <div className="text-sm font-medium text-gray-500">결혼 만족도</div>
                <Badge variant="secondary" className="text-lg px-3 py-1">
                  {filters.rate_marriage}/5
                </Badge>
              </div>
              <div className="space-y-2">
                <div className="text-sm font-medium text-gray-500">나이</div>
                <Badge variant="secondary" className="text-lg px-3 py-1">
                  {filters.age}세
                </Badge>
              </div>
              <div className="space-y-2">
                <div className="text-sm font-medium text-gray-500">결혼 기간</div>
                <Badge variant="secondary" className="text-lg px-3 py-1">
                  {filters.yrs_married}년
                </Badge>
              </div>
              <div className="space-y-2">
                <div className="text-sm font-medium text-gray-500">자녀 수</div>
                <Badge variant="secondary" className="text-lg px-3 py-1">
                  {filters.children}명
                </Badge>
              </div>
              <div className="space-y-2">
                <div className="text-sm font-medium text-gray-500">종교 성향</div>
                <Badge variant="secondary" className="text-lg px-3 py-1">
                  {filters.religious}/4
                </Badge>
              </div>
              <div className="space-y-2">
                <div className="text-sm font-medium text-gray-500">교육 수준</div>
                <Badge variant="secondary" className="text-lg px-3 py-1">
                  {filters.educ}
                </Badge>
              </div>
              <div className="space-y-2">
                <div className="text-sm font-medium text-gray-500">직업</div>
                <Badge variant="secondary" className="text-lg px-3 py-1">
                  {filters.occupation}
                </Badge>
              </div>
              <div className="space-y-2">
                <div className="text-sm font-medium text-gray-500">배우자 직업</div>
                <Badge variant="secondary" className="text-lg px-3 py-1">
                  {filters.occupation_husb}
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        <Separator />

        {/* Action Buttons */}
        <Card>
          <CardContent className="pt-6">
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Button onClick={handleCompare} variant="outline" size="lg">
                다른 조건으로 비교하기
              </Button>
              <Button onClick={handleNewTest} size="lg" className="px-8">
                새로 테스트하기
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Share Section */}
        <Card>
          <CardHeader>
            <CardTitle className="text-center text-2xl">결과 공유하기</CardTitle>
            <p className="text-center text-gray-600">
              분석 결과를 친구들과 공유해보세요
            </p>
          </CardHeader>
          {/* <CardContent className="text-center">
            <ShareButtons />
          </CardContent> */}
        </Card>

        <EthicsNotice className="mt-8" />
      </div>
    </div>
  );
}
