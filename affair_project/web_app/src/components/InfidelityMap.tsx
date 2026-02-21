"use client";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useThemeColors } from "../hooks/useThemeColors";

interface StateData {
  State: string;
  Rank: number;
  "Infidelity Rate": string;
  value: number;
  name: string;
}

interface InfidelityMapProps {
  showRankings?: boolean;
  onToggleRankings?: () => void;
  className?: string;
}

export function InfidelityMap({ showRankings = false, onToggleRankings, className }: InfidelityMapProps) {
  const { colors, isDark } = useThemeColors();
  const svgRef = useRef<SVGSVGElement>(null);
  const [data, setData] = useState<StateData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  // 마운트 상태 확인
  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;
    
    const loadData = async () => {
      try {
        setLoading(true);
        
        // API를 통해 실제 데이터 로드
        const csvData = await fetch('/api/infidelity-data').then(res => res.json());
        
        // 데이터 정리
        const processedData = csvData.map((row: Record<string, string | number>) => ({
          ...row,
          Rank: +row.Rank,
          value: +String(row['Infidelity Rate']).replace('%', ''),
          name: row.State
        })) as StateData[];
        
        setData(processedData);
        setLoading(false);
      } catch (err) {
        console.error('데이터 로드 실패:', err);
        setError('데이터를 불러오는데 실패했습니다.');
        setLoading(false);
      }
    };

    loadData();
  }, [mounted]);

  useEffect(() => {
    if (!mounted || !data.length || !svgRef.current) return;

    // D3.js 로드 확인
    if (typeof window !== 'undefined' && (window as unknown as Record<string, unknown>).d3) {
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
  }, [mounted, data]);

  const renderMap = async () => {
    try {
      const d3 = await import('d3');
      const topojson = await import('topojson-client');
      
      if (!svgRef.current || !data.length) return;

      const svg = d3.select(svgRef.current);
      const container = svgRef.current.parentElement;
      const width = container ? container.clientWidth - 32 : 800; // 패딩 고려
      const height = 600;
      
      // SVG 크기를 컨테이너에 맞춤
      svg.attr('width', width).attr('height', height);

      // 투영 설정 - 컨테이너에 꽉 차도록 조정
      const projection = d3.geoAlbersUsa()
        .translate([width / 2, height / 2])
        .scale(Math.min(width, height) * 0.9); // 스케일을 0.9로 증가
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
          return item ? color(item.value) : (isDark ? '#475569' : '#e2e8f0');
        })
        .attr('stroke', isDark ? colors.border : '#ffffff')
        .attr('stroke-width', 0.5)
        .style('cursor', 'pointer')
        .on('mouseenter', function(this: any, event: any, d: any) {
          const item = dataMap.get(d.properties.name);
          if (item) {
            d3.select(this).attr('stroke-width', 2);
            showTooltip(event, d.properties.name, item);
          }
        })
        .on('mouseleave', function(this: any) {
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

  // 서버 사이드 렌더링 시 로딩 상태 유지
  if (!mounted || loading) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center h-96">
          <div className="text-center">
            <div 
              className="animate-spin rounded-full h-12 w-12 border-b-2 mx-auto mb-4"
              style={{ borderColor: colors.brand.primary }}
            ></div>
            <p style={{ color: colors.text.quaternary }}>
              지도 데이터를 불러오는 중...
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center h-96">
          <div className="text-center" style={{ color: colors.brand.danger }}>
            <p className="text-lg font-semibold mb-2">오류가 발생했습니다</p>
            <p className="text-sm">{error}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <>
      <div className="space-y-4">
        {/* 범례 */}
        <div className="flex items-center gap-2 text-sm flex-wrap">
          <span 
            className="font-semibold transition-colors duration-300"
            style={{ color: colors.text.primary }}
          >
            Cheated(%)
          </span>
          {data.length > 0 && (
            <>
              {[0, 20, 40, 60, 80, 100].map((value) => {
                const color = value === 0 ? '#c7f9cc' : 
                              value === 100 ? '#0b3c8a' : 
                              `hsl(${200 - (value * 1.5)}, 70%, ${60 - (value * 0.3)}%)`;
                return (
                  <div key={value} className="flex items-center gap-1">
                    <div 
                      className="w-4 h-4 rounded-full border transition-all duration-300"
                      style={{ 
                        backgroundColor: color,
                        borderColor: colors.border
                      }}
                    />
                    <span style={{ color: colors.text.quinary }}>{value}%</span>
                  </div>
                );
              })}
            </>
          )}
        </div>

        {/* 지도 */}
        <div 
          className="rounded-lg p-4 h-[600px] w-full transition-all duration-300 flex justify-center items-center"
          style={{
            backgroundColor: isDark ? colors.background.icon : colors.background.primary,
            border: `1px solid ${colors.border}`,
            minHeight: '600px',
            width: '100%'
          }}
        >
          <svg 
            ref={svgRef}
            className="w-full h-[600px]"
            role="img" 
            aria-label="US choropleth map of infidelity rate"
            style={{ 
              width: '100%',
              height: '600px',
              display: 'block'
            }}
          />
        </div>

        {/* 순위표 토글 버튼 */}
        {onToggleRankings && (
          <div className="mt-6">
            <Button
              onClick={onToggleRankings}
              className="w-full transition-all duration-300 hover:scale-105"
              style={{
                backgroundColor: colors.background.button,
                color: colors.text.primary,
                border: `1px solid ${colors.border}`
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = colors.background.buttonHover;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = colors.background.button;
              }}
            >
              <span className="flex items-center gap-2">
                {showRankings ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                {showRankings ? '미국 주별 순위 숨기기' : '미국 주별 순위 보기'}
              </span>
            </Button>
          </div>
        )}

        {/* 순위표 */}
        {showRankings && (
          <div className="mt-6">
            <h3 className="text-lg font-semibold mb-3" style={{ color: colors.text.primary }}>
              주별 순위
            </h3>
            <div 
              className="rounded-lg overflow-hidden transition-all duration-300"
              style={{
                backgroundColor: isDark ? colors.background.icon : colors.background.primary,
                border: `1px solid ${colors.border}`
              }}
            >
              <table className="w-full">
                <thead 
                  className="transition-all duration-300"
                  style={{
                    backgroundColor: isDark ? colors.background.secondary : colors.background.tertiary
                  }}
                >
                  <tr>
                    <th className="text-left p-3 font-semibold" style={{ color: colors.text.primary }}>주</th>
                    <th className="text-left p-3 font-semibold" style={{ color: colors.text.primary }}>바람지수</th>
                    <th className="text-left p-3 font-semibold" style={{ color: colors.text.primary }}>순위</th>
                  </tr>
                </thead>
                <tbody>
                  {data
                    .sort((a, b) => a.Rank - b.Rank)
                    .slice(0, 10) // 상위 10개만 표시
                    .map((row, index) => (
                      <tr 
                        key={row.State} 
                        className="border-t transition-all duration-300 hover:bg-opacity-50"
                        style={{
                          borderColor: colors.border,
                          backgroundColor: 'transparent'
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor = isDark 
                            ? `${colors.background.secondary}80` 
                            : `${colors.background.tertiary}80`;
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = 'transparent';
                        }}
                      >
                        <td className="p-3" style={{ color: colors.text.primary }}>{row.State}</td>
                        <td className="p-3">
                          <Badge variant="secondary">
                            {(row.value.toFixed(1))}%
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
        )}
      </div>

      {/* 툴팁 */}
      <div
        id="tooltip"
        className="fixed pointer-events-none p-2 rounded-lg text-sm opacity-0 transition-opacity duration-200 z-50 border"
        style={{ 
          maxWidth: '200px',
          backgroundColor: isDark ? '#0b1220' : '#ffffff',
          borderColor: isDark ? '#1f2937' : '#e5e7eb',
          color: isDark ? '#e2e8f0' : '#1f2937'
        }}
      />
    </>
  );
}
