"use client";
import React from "react";
import { Badge } from "@/components/ui/badge";
import { motion } from "motion/react";
import { useStore } from "@/store/useStore";

interface ProgressBarProps {
  className?: string;
}

const steps = [
  { id: "landing", label: "시작하기", icon: "🏠" },
  { id: "filters", label: "입력하기", icon: "📝" },
  { id: "result", label: "결과보기", icon: "📊" },
  { id: "compare", label: "비교하기", icon: "⚖️" },
];

export function ProgressBar({ className }: ProgressBarProps) {
  const currentStep = useStore((state) => state.currentStep);

  const getCurrentStepIndex = () => {
    return steps.findIndex(step => step.id === currentStep);
  };

  const currentIndex = getCurrentStepIndex();

  return (
    <motion.div 
      className={`bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-40 ${className}`}
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
    >
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          {/* 로고/제목 */}
          <motion.div 
            className="flex items-center gap-3"
            initial={{ x: -50, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <h1 className="text-xl font-bold text-gray-900">바람지수 분석기</h1>
            <Badge variant="outline" className="text-xs">
              Beta
            </Badge>
          </motion.div>

          {/* 진행률 바 */}
          <div className="hidden md:flex items-center gap-6">
            {steps.map((step, index) => {
              const isActive = step.id === currentStep;
              const isCompleted = index < currentIndex;
              const isUpcoming = index > currentIndex;

              return (
                <motion.div 
                  key={step.id} 
                  className="flex items-center gap-2"
                  initial={{ y: 20, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ 
                    duration: 0.5, 
                    delay: 0.1 * index,
                    ease: "easeOut"
                  }}
                >
                  {/* 단계 아이콘 */}
                  <motion.div
                    className={`w-10 h-10 rounded-full flex items-center justify-center text-lg transition-all duration-300 ${
                      isActive
                        ? "bg-blue-500 text-white shadow-lg"
                        : isCompleted
                        ? "bg-green-500 text-white"
                        : "bg-gray-200 text-gray-500"
                    }`}
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.95 }}
                    animate={isActive ? { scale: [1, 1.1, 1] } : {}}
                    transition={{ duration: 0.3 }}
                  >
                    {step.icon}
                  </motion.div>

                  {/* 단계 라벨 */}
                  <div className="text-sm">
                    <div
                      className={`font-medium ${
                        isActive ? "text-blue-600" : isCompleted ? "text-green-600" : "text-gray-500"
                      }`}
                    >
                      {step.label}
                    </div>
                    {isActive && (
                      <motion.div 
                        className="text-xs text-blue-500 mt-1"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.3 }}
                      >
                        현재 단계
                      </motion.div>
                    )}
                  </div>

                  {/* 연결선 */}
                  {index < steps.length - 1 && (
                    <motion.div
                      className={`w-8 h-0.5 transition-colors duration-300 ${
                        isCompleted ? "bg-green-400" : "bg-gray-200"
                      }`}
                      initial={{ scaleX: 0 }}
                      animate={{ scaleX: 1 }}
                      transition={{ duration: 0.5, delay: 0.2 * index }}
                    />
                  )}
                </motion.div>
              );
            })}
          </div>

          {/* 모바일용 간단한 표시 */}
          <motion.div 
            className="md:hidden"
            initial={{ x: 50, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <Badge variant="secondary">
              {steps[currentIndex]?.label} ({currentIndex + 1}/{steps.length})
            </Badge>
          </motion.div>
        </div>

        {/* 모바일용 진행률 바 */}
        <motion.div 
          className="md:hidden mt-3"
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <div className="w-full bg-gray-200 rounded-full h-2">
            <motion.div
              className="bg-blue-500 h-2 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${((currentIndex + 1) / steps.length) * 100}%` }}
              transition={{ duration: 0.8, ease: "easeOut" }}
            />
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
}
