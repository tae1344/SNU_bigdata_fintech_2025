"use client";
import React, { useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface StateData {
  State: string;
  Rank: number;
  "Infidelity Rate": string;
  value: number;
  name: string;
}

interface InfidelityMapProps {
  className?: string;
}

export function InfidelityMap({ className }: InfidelityMapProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [data, setData] = useState<StateData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        
        // CSV 데이터 로드 (실제 프로젝트에서는 API나 정적 파일에서 로드)
        const csvData = await fetch('/api/infidelity-data').then(res => res.json());
        
        // 데이터 정리
        const processedData = csvData.map((row: any) => ({
          ...row,
          Rank: +row.Rank,
          value: +String(row['Infidelity Rate']).replace('%', ''),
          name: row.State
        }));
        
        setData(processedData);
        setLoading(false);
      } catch (err) {
        console.error('데이터 로드 실패:', err);
        setError('데이터를 불러오는데 실패했습니다.');
        setLoading(false);
      }
    };

    loadData();
  }, []);

  useEffect(() => {
    if (!data.length || !svgRef.current) return;

    // D3.js 로드 확인
    if (typeof window !== 'undefined' && (window as any).d3) {
      renderMap();
    } else {
      // D3.js가 로드되지 않은 경우 동적 로드
      const loadD3 = async () => {
        try {
          await import('d3');
          await import('topojson-client');
          renderMap();
        } catch (err) {
          console.error('D3.js 로드 실패:', err);
          setError('차트 라이브러리를 로드하는데 실패했습니다.');
        }
      };
      loadD3();
    }
  }, [data]);

  const renderMap = async () => {
    try {
      const d3 = await import('d3');
      const topojson = await import('topojson-client');
      
      if (!svgRef.current || !data.length) return;

      const svg = d3.select(svgRef.current);
      const width = svgRef.current.clientWidth;
      const height = 600;
      
      svg.attr('width', width).attr('height', height);

      // 투영 설정
      const projection = d3.geoAlbersUsa()
        .translate([width / 2, height / 2])
        .scale(1000);
      const path = d3.geoPath(projection);

      // 색상 스케일
      const minV = d3.min(data, d => d.value) || 0;
      const maxV = d3.max(data, d => d.value) || 100;
      const color = d3.scaleSequential()
        .domain([minV, maxV])
        .interpolator(d3.interpolateRgb('#c7f9cc', '#0b3c8a'));

      // 미국 지도 데이터 로드
      const usResponse = await fetch('https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json');
      const us = await usResponse.json();
      
      const states = topojson.feature(us, us.objects.states) as any;
      const dataMap = new Map(data.map(d => [d.name, d]));

      // 기존 경로 제거
      svg.selectAll('path.state').remove();

      // 주 경로 그리기
      svg.selectAll('path.state')
        .data(states.features)
        .join('path')
        .attr('class', 'state')
        .attr('d', (d: any) => path(d) || '')
        .attr('fill', (d: any) => {
          const item = dataMap.get(d.properties.name);
          return item ? color(item.value) : '#cfd4dc';
        })
        .attr('stroke', 'white')
        .attr('stroke-width', 0.5)
        .style('cursor', 'pointer')
        .on('mouseenter', function(event: any, d: any) {
          const item = dataMap.get(d.properties.name);
          if (item) {
            d3.select(this).attr('stroke-width', 2);
            showTooltip(event, d.properties.name, item);
          }
        })
        .on('mouseleave', function() {
          d3.select(this).attr('stroke-width', 0.5);
          hideTooltip();
        });

    } catch (err) {
      console.error('지도 렌더링 실패:', err);
      setError('지도를 렌더링하는데 실패했습니다.');
    }
  };

  const showTooltip = (event: any, stateName: string, data: StateData) => {
    const tooltip = document.getElementById('tooltip');
    if (tooltip) {
      tooltip.innerHTML = `
        <strong>${stateName}</strong><br/>
        순위: #${data.Rank}<br/>
        <strong>${data.value}%</strong>
      `;
      tooltip.style.left = (event.clientX + 12) + 'px';
      tooltip.style.top = (event.clientY + 12) + 'px';
      tooltip.style.opacity = '1';
    }
  };

  const hideTooltip = () => {
    const tooltip = document.getElementById('tooltip');
    if (tooltip) {
      tooltip.style.opacity = '0';
    }
  };

  if (loading) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <p className="text-gray-600">지도 데이터를 불러오는 중...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center h-96">
          <div className="text-center text-red-500">
            <p className="text-lg font-semibold mb-2">오류가 발생했습니다</p>
            <p className="text-sm">{error}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <>
      <Card className={className}>
        <CardHeader>
          <CardTitle className="text-2xl font-bold">🇺🇸 미국 주별 바람지수</CardTitle>
          <p className="text-gray-600">
            각 주별 바람 지수(자기보고형 설문 %). 파란색이 진할수록 비율이 높습니다.
          </p>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* 범례 */}
            <div className="flex items-center gap-2 text-sm text-gray-600 flex-wrap">
              <span className="font-semibold text-gray-900">{"Cheated(%)"}</span>
              {data.length > 0 && (
                <>
                  {[0, 20, 40, 60, 80, 100].map((value) => {
                    const color = value === 0 ? '#c7f9cc' : 
                                 value === 100 ? '#0b3c8a' : 
                                 `hsl(${200 - (value * 1.5)}, 70%, ${60 - (value * 0.3)}%)`;
                    return (
                      <div key={value} className="flex items-center gap-1">
                        <div 
                          className="w-4 h-4 rounded-full border border-gray-300"
                          style={{ backgroundColor: color }}
                        />
                        <span>{value}%</span>
                      </div>
                    );
                  })}
                </>
              )}
            </div>

            {/* 지도 */}
            <div className="bg-gray-50 rounded-lg p-4">
              <svg 
                ref={svgRef}
                className="w-full h-[600px]"
                role="img" 
                aria-label="US choropleth by infidelity rate"
              />
            </div>

            {/* 순위표 */}
            <div className="mt-6">
              <h3 className="text-lg font-semibold mb-3">주별 순위</h3>
              <div className="bg-gray-50 rounded-lg overflow-hidden">
                <table className="w-full">
                  <thead className="bg-gray-100">
                    <tr>
                      <th className="text-left p-3 font-semibold text-gray-700">주</th>
                      <th className="text-left p-3 font-semibold text-gray-700">바람지수</th>
                      <th className="text-left p-3 font-semibold text-gray-700">순위</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data
                      .sort((a, b) => a.Rank - b.Rank)
                      .slice(0, 10) // 상위 10개만 표시
                      .map((row, index) => (
                        <tr key={row.State} className="border-t border-gray-200 hover:bg-gray-50">
                          <td className="p-3">{row.State}</td>
                          <td className="p-3">
                            <Badge variant="secondary">
                              {Math.round(row.value)}%
                            </Badge>
                          </td>
                          <td className="p-3">
                            <Badge variant={index < 3 ? "default" : "outline"}>
                              #{row.Rank}
                            </Badge>
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 툴팁 */}
      <div
        id="tooltip"
        className="fixed pointer-events-none bg-gray-900 text-white p-2 rounded-lg text-sm opacity-0 transition-opacity duration-200 z-50 border border-gray-700"
        style={{ maxWidth: '200px' }}
      />
    </>
  );
}
