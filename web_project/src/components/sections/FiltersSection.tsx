"use client";
import React from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { useStore } from "@/store/useStore";
import { calculateScore } from "@/lib/scoring";
import { EthicsNotice } from "@/components/legacy/EthicsNotice";
import { Filters } from "@/types";

export function FiltersSection() {
  const { filters, updateFilter, setResult, setCurrentStep, setLoading } = useStore();

  const fields = [
    {
      key: "rate_marriage" as const,
      label: "결혼 만족도",
      min: 1,
      max: 5,
      type: "radio" as const,
      description: "해당하는 항목을 선택하세요",
      options: [
        { value: 1, label: "매우 불만족" },
        { value: 2, label: "불만족" },
        { value: 3, label: "보통" },
        { value: 4, label: "만족" },
        { value: 5, label: "매우 만족" }
      ]
    },
    {
      key: "age" as const,
      label: "나이",
      min: 18,
      max: 80,
      type: "slider" as const,
      description: "18세 ~ 80세"
    },
    {
      key: "yrs_married" as const,
      label: "결혼 기간",
      min: 0,
      max: 60,
      type: "slider" as const,
      description: "0년 ~ 60년"
    },
    {
      key: "children" as const,
      label: "자녀 수",
      min: 0,
      max: 10,
      type: "slider" as const,
      description: "0명 ~ 10명"
    },
    {
      key: "religious" as const,
      label: "종교 성향",
      min: 1,
      max: 4,
      type: "radio" as const,
      description: "해당하는 항목을 선택하세요",
      options: [
        { value: 1, label: "무종교" },
        { value: 2, label: "덜 종교적" },
        { value: 3, label: "보통" },
        { value: 4, label: "매우 종교적" }
      ]
    },
    {
      key: "educ" as const,
      label: "교육 수준",
      min: 1,
      max: 6,
      type: "slider" as const,
      description: "1: 초등학교, 6: 대학원"
    },
    {
      key: "occupation" as const,
      label: "직업 코드",
      min: 1,
      max: 6,
      type: "slider" as const,
      description: "1: 학생, 6: 전문직"
    },
    {
      key: "occupation_husb" as const,
      label: "배우자 직업 코드",
      min: 1,
      max: 6,
      type: "slider" as const,
      description: "1: 학생, 6: 전문직"
    },
  ];

  function update<K extends keyof typeof filters>(key: K, value: Filters[K]) {
    updateFilter(key, value);
  }

  function handleSubmit() {
    const result = calculateScore(filters);
    setResult(result);
    setLoading(true);
    setCurrentStep("result");
  }

  function handleBack() {
    setCurrentStep("landing");
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen px-4 py-8">
      <div className="w-full max-w-4xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-3xl font-bold text-gray-900">정보 입력</h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            아래 항목들을 입력해주세요. 모든 항목을 입력한 후 결과를 확인할 수 있습니다.
          </p>
        </div>

        {/* Form */}
        <form onSubmit={(e) => { e.preventDefault(); handleSubmit(); }} className="space-y-6">
          {/* Basic Info Section */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <span className="inline-flex items-center justify-center w-6 h-6 text-sm font-medium text-white bg-blue-600 rounded-full">1</span>
                기본 정보
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {fields.slice(0, 3).map((field) => (
                  <div key={field.key} className="space-y-3">
                    <Label htmlFor={field.key} className="text-sm font-medium">
                      {field.label}
                    </Label>
                    <p className="text-xs text-muted-foreground">{field.description}</p>
                    
                    {field.type === "radio" ? (
                      <RadioGroup
                        value={String(filters[field.key])}
                        onValueChange={(value) => update(field.key, Number(value) as Filters[typeof field.key])}
                        required
                      >
                        {field.options?.map((option) => (
                          <div key={option.value} className="flex items-center space-x-2">
                            <RadioGroupItem value={String(option.value)} id={`${field.key}-${option.value}`} />
                            <Label htmlFor={`${field.key}-${option.value}`} className="text-sm">
                              {option.value}: {option.label}
                            </Label>
                          </div>
                        ))}
                      </RadioGroup>
                    ) : (
                      <div className="space-y-3">
                        <Slider
                          id={field.key}
                          min={field.min}
                          max={field.max}
                          step={1}
                          value={[filters[field.key]]}
                          onValueChange={(values) => update(field.key, values[0] as Filters[typeof field.key])}
                          className="w-full"
                        />
                        <div className="flex justify-between text-sm text-muted-foreground">
                          <span>{field.min}</span>
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                            {filters[field.key]}
                          </span>
                          <span>{field.max}</span>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Family & Values Section */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <span className="inline-flex items-center justify-center w-6 h-6 text-sm font-medium text-white bg-green-600 rounded-full">2</span>
                가정/가치관
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {fields.slice(3, 5).map((field) => (
                  <div key={field.key} className="space-y-3">
                    <Label htmlFor={field.key} className="text-sm font-medium">
                      {field.label}
                    </Label>
                    <p className="text-xs text-muted-foreground">{field.description}</p>
                    
                    {field.type === "radio" ? (
                      <RadioGroup
                        value={String(filters[field.key])}
                        onValueChange={(value) => update(field.key, Number(value) as Filters[typeof field.key])}
                        required
                      >
                        {field.options?.map((option) => (
                          <div key={option.value} className="flex items-center space-x-2">
                            <RadioGroupItem value={String(option.value)} id={`${field.key}-${option.value}`} />
                            <Label htmlFor={`${field.key}-${option.value}`} className="text-sm">
                              {option.value}: {option.label}
                            </Label>
                          </div>
                        ))}
                      </RadioGroup>
                    ) : (
                      <div className="space-y-3">
                        <Slider
                          id={field.key}
                          min={field.min}
                          max={field.max}
                          step={1}
                          value={[filters[field.key]]}
                          onValueChange={(values) => update(field.key, values[0] as Filters[typeof field.key])}
                          className="w-full"
                        />
                        <div className="flex justify-between text-sm text-muted-foreground">
                          <span>{field.min}</span>
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                            {filters[field.key]}
                          </span>
                          <span>{field.max}</span>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Education & Occupation Section */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <span className="inline-flex items-center justify-center w-6 h-6 text-sm font-medium text-white bg-purple-600 rounded-full">3</span>
                교육/직업
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {fields.slice(5).map((field) => (
                  <div key={field.key} className="space-y-3">
                    <Label htmlFor={field.key} className="text-sm font-medium">
                      {field.label}
                    </Label>
                    <p className="text-xs text-muted-foreground">{field.description}</p>
                    
                    <div className="space-y-3">
                      <Slider
                        id={field.key}
                        min={field.min}
                        max={field.max}
                        step={1}
                        value={[filters[field.key]]}
                        onValueChange={(values) => update(field.key, values[0] as Filters[typeof field.key])}
                        className="w-full"
                      />
                      <div className="flex justify-between text-sm text-muted-foreground">
                        <span>{field.min}</span>
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                          {filters[field.key]}
                        </span>
                        <span>{field.max}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <div className="border-t pt-6" />

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center pt-4">
            <Button
              type="button"
              variant="outline"
              onClick={handleBack}
              size="lg"
            >
              뒤로 가기
            </Button>
            
            <Button
              type="submit"
              size="lg"
              className="px-8"
            >
              결과 보기
            </Button>
          </div>
        </form>

        <EthicsNotice className="mt-8" />
      </div>
    </div>
  );
}
