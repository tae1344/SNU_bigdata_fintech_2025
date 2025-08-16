"use client";
import React, { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { motion } from "motion/react";

interface DataPoint {
  rate_marriage: number;
  age: number;
  yrs_married: number;
  children: number;
  religious: number;
  educ: number;
  occupation: number;
  occupation_husb: number;
  affairs: number;
}

interface StatisticsDashboardProps {
  className?: string;
}

export function StatisticsDashboard({ className }: StatisticsDashboardProps) {
  const [data, setData] = useState<DataPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    totalRecords: 0,
    averageAge: 0,
    averageMarriageSatisfaction: 0,
    averageChildren: 0,
    averageReligious: 0,
    averageEducation: 0,
    averageAffairs: 0,
    ageGroups: {} as Record<string, number>,
    marriageSatisfactionGroups: {} as Record<string, number>,
    childrenGroups: {} as Record<string, number>,
    religiousGroups: {} as Record<string, number>
  });

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        
        // CSV 데이터 로드 (실제 프로젝트에서는 API나 정적 파일에서 로드)
        const response = await fetch('/api/affairs-data');
        const csvData = await response.json();
        
        setData(csvData);
        
        // 통계 계산
        calculateStatistics(csvData);
        setLoading(false);
      } catch (err) {
        console.error('데이터 로드 실패:', err);
        setLoading(false);
      }
    };

    loadData();
  }, []);

  const calculateStatistics = (data: DataPoint[]) => {
    const totalRecords = data.length;
    const averageAge = data.reduce((sum, d) => sum + d.age, 0) / totalRecords;
    const averageMarriageSatisfaction = data.reduce((sum, d) => sum + d.rate_marriage, 0) / totalRecords;
    const averageChildren = data.reduce((sum, d) => sum + d.children, 0) / totalRecords;
    const averageReligious = data.reduce((sum, d) => sum + d.religious, 0) / totalRecords;
    const averageEducation = data.reduce((sum, d) => sum + d.educ, 0) / totalRecords;
    const averageAffairs = data.reduce((sum, d) => sum + d.affairs, 0) / totalRecords;

    // 연령대별 그룹화
    const ageGroups = data.reduce((acc, d) => {
      const ageGroup = d.age < 25 ? '18-24' : d.age < 35 ? '25-34' : d.age < 45 ? '35-44' : '45+';
      acc[ageGroup] = (acc[ageGroup] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    // 결혼 만족도별 그룹화
    const marriageSatisfactionGroups = data.reduce((acc, d) => {
      acc[d.rate_marriage.toString()] = (acc[d.rate_marriage.toString()] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    // 자녀 수별 그룹화
    const childrenGroups = data.reduce((acc, d) => {
      const childrenGroup = d.children === 0 ? '0명' : d.children === 1 ? '1명' : d.children === 2 ? '2명' : '3명+';
      acc[childrenGroup] = (acc[childrenGroup] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    // 종교성별 그룹화
    const religiousGroups = data.reduce((acc, d) => {
      acc[d.religious.toString()] = (acc[d.religious.toString()] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    setStats({
      totalRecords,
      averageAge: Math.round(averageAge * 10) / 10,
      averageMarriageSatisfaction: Math.round(averageMarriageSatisfaction * 100) / 100,
      averageChildren: Math.round(averageChildren * 100) / 100,
      averageReligious: Math.round(averageReligious * 100) / 100,
      averageEducation: Math.round(averageEducation * 100) / 100,
      averageAffairs: Math.round(averageAffairs * 100) / 100,
      ageGroups,
      marriageSatisfactionGroups,
      childrenGroups,
      religiousGroups
    });
  };

  if (loading) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <p className="text-gray-600">통계 데이터를 불러오는 중...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className={className}>
      {/* 전체 통계 요약 */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-2xl font-bold text-center">📊 데이터셋 통계 요약</CardTitle>
          <p className="text-center text-gray-600">
            총 {stats.totalRecords.toLocaleString()}개의 데이터 포인트 분석 결과
          </p>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <motion.div 
              className="text-center p-4 bg-blue-50 rounded-lg"
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              <div className="text-2xl font-bold text-blue-600">{stats.averageAge}세</div>
              <div className="text-sm text-gray-600">평균 나이</div>
            </motion.div>
            
            <motion.div 
              className="text-center p-4 bg-green-50 rounded-lg"
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <div className="text-2xl font-bold text-green-600">{stats.averageMarriageSatisfaction}/5</div>
              <div className="text-sm text-gray-600">평균 결혼 만족도</div>
            </motion.div>
            
            <motion.div 
              className="text-center p-4 bg-purple-50 rounded-lg"
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.3 }}
            >
              <div className="text-2xl font-bold text-purple-600">{stats.averageChildren}명</div>
              <div className="text-sm text-gray-600">평균 자녀 수</div>
            </motion.div>
            
            <motion.div 
              className="text-center p-4 bg-orange-50 rounded-lg"
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.4 }}
            >
              <div className="text-2xl font-bold text-orange-600">{stats.averageReligious}/4</div>
              <div className="text-sm text-gray-600">평균 종교성</div>
            </motion.div>
          </div>
        </CardContent>
      </Card>

      {/* 상세 분포 차트 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 연령대별 분포 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span className="text-xl">👥</span>
              연령대별 분포
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {Object.entries(stats.ageGroups).map(([group, count], index) => (
                <motion.div 
                  key={group}
                  className="flex items-center justify-between"
                  initial={{ x: -50, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  transition={{ duration: 0.5, delay: 0.1 * index }}
                >
                  <span className="font-medium">{group}</span>
                  <div className="flex items-center gap-2">
                    <div className="w-32 bg-gray-200 rounded-full h-2">
                      <motion.div
                        className="bg-blue-500 h-2 rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${(count / stats.totalRecords) * 100}%` }}
                        transition={{ duration: 0.8, delay: 0.1 * index }}
                      />
                    </div>
                    <Badge variant="secondary">{count}명</Badge>
                  </div>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* 결혼 만족도별 분포 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span className="text-xl">💕</span>
              결혼 만족도별 분포
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {Object.entries(stats.marriageSatisfactionGroups)
                .sort(([a], [b]) => parseInt(a) - parseInt(b))
                .map(([satisfaction, count], index) => (
                <motion.div 
                  key={satisfaction}
                  className="flex items-center justify-between"
                  initial={{ x: 50, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  transition={{ duration: 0.5, delay: 0.1 * index }}
                >
                  <span className="font-medium">{satisfaction}점</span>
                  <div className="flex items-center gap-2">
                    <div className="w-32 bg-gray-200 rounded-full h-2">
                      <motion.div
                        className="bg-green-500 h-2 rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${(count / stats.totalRecords) * 100}%` }}
                        transition={{ duration: 0.8, delay: 0.1 * index }}
                      />
                    </div>
                    <Badge variant="secondary">{count}명</Badge>
                  </div>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* 자녀 수별 분포 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span className="text-xl">👶</span>
              자녀 수별 분포
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {Object.entries(stats.childrenGroups)
                .sort(([a], [b]) => {
                  const aNum = a === '0명' ? 0 : a === '1명' ? 1 : a === '2명' ? 2 : 3;
                  const bNum = b === '0명' ? 0 : b === '1명' ? 1 : b === '2명' ? 2 : 3;
                  return aNum - bNum;
                })
                .map(([children, count], index) => (
                <motion.div 
                  key={children}
                  className="flex items-center justify-between"
                  initial={{ y: 20, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ duration: 0.5, delay: 0.1 * index }}
                >
                  <span className="font-medium">{children}</span>
                  <div className="flex items-center gap-2">
                    <div className="w-32 bg-gray-200 rounded-full h-2">
                      <motion.div
                        className="bg-purple-500 h-2 rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${(count / stats.totalRecords) * 100}%` }}
                        transition={{ duration: 0.8, delay: 0.1 * index }}
                      />
                    </div>
                    <Badge variant="secondary">{count}명</Badge>
                  </div>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* 종교성별 분포 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span className="text-xl">⛪</span>
              종교성별 분포
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {Object.entries(stats.religiousGroups)
                .sort(([a], [b]) => parseInt(a) - parseInt(b))
                .map(([religious, count], index) => (
                <motion.div 
                  key={religious}
                  className="flex items-center justify-between"
                  initial={{ y: 20, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ duration: 0.5, delay: 0.1 * index }}
                >
                  <span className="font-medium">{religious}점</span>
                  <div className="flex items-center gap-2">
                    <div className="w-32 bg-gray-200 rounded-full h-2">
                      <motion.div
                        className="bg-orange-500 h-2 rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${(count / stats.totalRecords) * 100}%` }}
                        transition={{ duration: 0.8, delay: 0.1 * index }}
                      />
                    </div>
                    <Badge variant="secondary">{count}명</Badge>
                  </div>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
