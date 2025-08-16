"use client";
import React, { useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useThemeColors } from "../hooks/useThemeColors";

interface CountryData {
  name: string;
  value: number;
  rank: number;
}

interface WorldInfidelityMapProps {
  showRankings?: boolean;
  onToggleRankings?: () => void;
  className?: string;
}

export function WorldInfidelityMap({ showRankings = false, onToggleRankings, className }: WorldInfidelityMapProps) {
  const { colors, isDark } = useThemeColors();
  const svgRef = useRef<SVGSVGElement>(null);
  const [data, setData] = useState<Map<string, CountryData>>(new Map());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  // 국가 데이터 (ISO3 코드 -> 데이터)
  const countryData = {
    // Europe
    FRA: { name: 'France', value: 43, rank: 5 },
    DEU: { name: 'Germany', value: 34, rank: 18 },
    SWE: { name: 'Sweden', value: 45, rank: 3 },
    NOR: { name: 'Norway', value: 37, rank: 12 },
    ESP: { name: 'Spain', value: 38, rank: 10 },
    ITA: { name: 'Italy', value: 36, rank: 14 },
    CZE: { name: 'Czechia', value: 30, rank: 25 },
    UKR: { name: 'Ukraine', value: 28, rank: 30 },
    GBR: { name: 'United Kingdom', value: 32, rank: 21 },
    IRL: { name: 'Ireland', value: 26, rank: 34 },
    POL: { name: 'Poland', value: 27, rank: 32 },
    NLD: { name: 'Netherlands', value: 33, rank: 20 },
    BEL: { name: 'Belgium', value: 35, rank: 16 },

    // Americas
    USA: { name: 'United States', value: 36, rank: 15 },
    CAN: { name: 'Canada', value: 33, rank: 19 },
    MEX: { name: 'Mexico', value: 28, rank: 29 },
    BRA: { name: 'Brazil', value: 31, rank: 23 },
    ARG: { name: 'Argentina', value: 22, rank: 40 },
    CHL: { name: 'Chile', value: 25, rank: 35 },

    // Asia / Oceania
    THA: { name: 'Thailand', value: 50, rank: 1 },
    AUS: { name: 'Australia', value: 29, rank: 27 },
    NZL: { name: 'New Zealand', value: 30, rank: 24 },
    KOR: { name: 'South Korea', value: 24, rank: 37 },
    JPN: { name: 'Japan', value: 23, rank: 39 },
    VNM: { name: 'Vietnam', value: 21, rank: 41 },
  };

  // ISO3 -> ISO2 (국기용)
  const ISO3_TO_A2: Record<string, string> = {
    FRA: 'fr', DEU: 'de', SWE: 'se', NOR: 'no', ESP: 'es', ITA: 'it', CZE: 'cz', UKR: 'ua', GBR: 'gb', IRL: 'ie', POL: 'pl', NLD: 'nl', BEL: 'be',
    USA: 'us', CAN: 'ca', MEX: 'mx', BRA: 'br', ARG: 'ar', CHL: 'cl',
    THA: 'th', AUS: 'au', NZL: 'nz', KOR: 'kr', JPN: 'jp', VNM: 'vn'
  };

  // 숫자 ID -> ISO3 매핑
  const NUM_TO_ISO3: Record<number, string> = {
    250: 'FRA', 276: 'DEU', 752: 'SWE', 578: 'NOR', 724: 'ESP', 380: 'ITA', 203: 'CZE', 826: 'GBR', 372: 'IRL', 616: 'POL', 528: 'NLD', 56: 'BEL',
    840: 'USA', 124: 'CAN', 484: 'MEX', 76: 'BRA', 32: 'ARG', 152: 'CHL', 764: 'THA', 36: 'AUS', 554: 'NZL', 410: 'KOR', 392: 'JPN', 704: 'VNM', 804: 'UKR'
  };

  const ISO3_from_numeric = (n: number): string | null => {
    return NUM_TO_ISO3[n] || null;
  };

  // 마운트 상태 확인
  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;
    
    setData(new Map(Object.entries(countryData)));
    setLoading(false);
  }, [mounted]);

  useEffect(() => {
    if (!mounted || !data.size || !svgRef.current) return;

    const loadD3 = async () => {
      try {
        const d3 = await import('d3');
        const topojson = await import('topojson-client');
        renderMap(d3, topojson);
      } catch (err) {
        console.error('D3.js 로드 실패:', err);
        setError('차트 라이브러리를 로드하는데 실패했습니다.');
      }
    };

    loadD3();

    // ResizeObserver 추가하여 컨테이너 크기 변화 감지
    const resizeObserver = new ResizeObserver(() => {
      if (data.size && svgRef.current) {
        loadD3();
      }
    });

    if (svgRef.current.parentElement) {
      resizeObserver.observe(svgRef.current.parentElement);
    }

    return () => {
      resizeObserver.disconnect();
    };
  }, [mounted, data]);

  const renderMap = async (d3: any, topojson: any) => {
    try {
      if (!svgRef.current || !data.size) return;

      const svg = d3.select(svgRef.current);
      const container = svgRef.current.parentElement;
      const width = container ? container.clientWidth : 800;
      const height = container ? container.clientHeight : 600;

      // SVG 크기를 컨테이너에 맞춤
      svg.attr('width', width).attr('height', height);

      // 투영 설정 - 컨테이너에 꽉 차도록 마진 최소화
      const projection = d3.geoNaturalEarth1()
        .fitExtent([[20, 20], [width - 20, height - 20]], { type: 'Sphere' });
      const path = d3.geoPath(projection);

      // 색상 스케일 (10% -> 55%)
      const minV = 10, maxV = 55;
      const color = d3.scaleSequential()
        .domain([minV, maxV])
        .interpolator(d3.interpolateRgb('#c7f9cc', '#0b3c8a'));

      // 세계 지도 데이터 로드
      const worldResponse = await fetch('https://unpkg.com/world-atlas@2/countries-110m.json');
      const world = await worldResponse.json();
      const countries = topojson.feature(world, world.objects.countries);

      // 기존 내용 제거
      svg.selectAll('*').remove();

      // 구체 배경 그리기
      svg.append('path')
        .attr('d', path({ type: 'Sphere' }))
        .attr('fill', isDark ? colors.background.primary : '#f8fafc');

      // 국가 그룹
      const g = svg.append('g');
      const tooltip = d3.select('#world-tooltip');

      // 남극대륙 제외
      const filtered = countries.features.filter((f: any) => 
        f.properties && f.properties.name !== 'Antarctica'
      );

      // 국가 경로 그리기
      g.selectAll('path.country')
        .data(filtered)
        .join('path')
        .attr('class', 'country')
        .attr('d', path)
        .attr('fill', (d: any) => {
          const code = d.id ? ISO3_from_numeric(d.id) : null;
          const item = code && data.get(code);
          return item ? color(item.value) : (isDark ? '#475569' : '#e2e8f0');
        })
        .attr('stroke', isDark ? colors.border : '#cbd5e1')
        .attr('stroke-width', 0.5)
        .style('cursor', 'pointer')
        .on('mousemove', function(this: any, event: any, d: any) {
          const [x, y] = d3.pointer(event);
          const code = d.id ? ISO3_from_numeric(d.id) : null;
          const item = code && data.get(code);
          const name = item?.name || (d.properties?.name as string) || 'Not ranked';

          d3.select(this)
            .attr('stroke', colors.brand.primary)
            .attr('stroke-width', 1.2)
            .attr('opacity', 0.95);

          if (item) {
            showTooltip(event, name, item);
          }
        })
        .on('mouseleave', function(this: any) {
          d3.select(this)
            .attr('stroke', isDark ? colors.border : '#cbd5e1')
            .attr('stroke-width', 0.5)
            .attr('opacity', 1);
          hideTooltip();
        });

    } catch (err) {
      console.error('지도 렌더링 실패:', err);
      setError('지도를 렌더링하는데 실패했습니다.');
    }
  };

  const showTooltip = (event: any, countryName: string, data: CountryData) => {
    const tooltip = document.getElementById('world-tooltip');
    if (tooltip) {
      tooltip.innerHTML = `
        <strong>${countryName}</strong><br/>
        순위: #${data.rank}<br/>
        <strong>${data.value}%</strong>
      `;
      tooltip.style.left = (event.clientX + 12) + 'px';
      tooltip.style.top = (event.clientY + 12) + 'px';
      tooltip.style.opacity = '1';
      tooltip.style.transform = 'translateY(0)';
    }
  };

  const hideTooltip = () => {
    const tooltip = document.getElementById('world-tooltip');
    if (tooltip) {
      tooltip.style.opacity = '0';
      tooltip.style.transform = 'translateY(4px)';
    }
  };

  // 서버 사이드 렌더링 시 로딩 상태 유지
  if (!mounted || loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div 
            className="animate-spin rounded-full h-12 w-12 border-b-2 mx-auto mb-4"
            style={{ borderColor: colors.brand.primary }}
          ></div>
          <p style={{ color: colors.text.quaternary }}>
            세계 지도 데이터를 불러오는 중...
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center" style={{ color: colors.brand.danger }}>
          <p className="text-lg font-semibold mb-2">오류가 발생했습니다</p>
          <p className="text-sm">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <>
      {/* 범례 */}
      <div className="flex items-center gap-2 text-sm flex-wrap mb-4">
        <span 
          className="font-semibold transition-colors duration-300"
          style={{ color: colors.text.primary }}
        >
          Cheated(%)
        </span>
        {[10, 15, 20, 25, 30, 35, 40, 45, 50, 55].map((value) => {
          const color = value === 10 ? '#c7f9cc' : 
                       value === 55 ? '#0b3c8a' : 
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
      </div>

      {/* 지도 */}
      <div 
        className="rounded-lg p-4 h-[600px] w-full transition-all duration-300 flex justify-center items-center"
        style={{
          backgroundColor: isDark ? colors.background.icon : colors.background.primary,
          border: `1px solid ${colors.border}`,
          minHeight: '600px'
        }}
      >
        <div className="w-full h-full flex justify-center items-center">
          <svg 
            ref={svgRef}
            className="w-full h-full max-w-full"
            role="img" 
            aria-label="World choropleth map of infidelity rates"
            style={{ 
              minWidth: '100%',
              maxWidth: '100%',
              minHeight: '600px'
            }}
          />
        </div>
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
              {showRankings ? '세계 순위 숨기기' : '세계 순위 보기'}
            </span>
          </Button>
        </div>
      )}

      {/* 순위표 */}
      {showRankings && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-3" style={{ color: colors.text.primary }}>
            세계 국가별 순위
          </h3>
          <div 
            className="rounded-lg overflow-hidden transition-all duration-300"
            style={{
              backgroundColor: isDark ? colors.background.icon : colors.background.primary,
              border: `1px solid ${colors.border}`
            }}
          >
            <table className="w-full">
              <thead>
                <tr style={{ backgroundColor: colors.background.tertiary }}>
                  <th className="text-left p-3 font-semibold transition-colors duration-300" style={{ color: colors.text.primary }}>
                    국가
                  </th>
                  <th className="text-left p-3 font-semibold transition-colors duration-300" style={{ color: colors.text.primary }}>
                    바람지수
                  </th>
                </tr>
              </thead>
              <tbody>
                {Array.from(data.values())
                  .sort((a, b) => a.rank - b.rank)
                  .slice(0, 15) // 상위 15개만 표시
                  .map((country, index) => {
                    const iso2 = ISO3_TO_A2[Object.keys(countryData).find(key => countryData[key as keyof typeof countryData].name === country.name) || ''];
                    return (
                      <tr 
                        key={country.name} 
                        className="border-t transition-all duration-300 hover:scale-105"
                        style={{ borderColor: colors.border }}
                      >
                        <td className="p-3 transition-colors duration-300" style={{ color: colors.text.primary }}>
                          <div className="flex items-center gap-3">
                            {iso2 && (
                              <img 
                                className="w-5 h-4 rounded border border-gray-300"
                                src={`https://flagcdn.com/w20/${iso2}.png`}
                                alt={`${country.name} flag`}
                                loading="lazy"
                              />
                            )}
                            <span>{country.name}</span>
                          </div>
                        </td>
                        <td className="p-3">
                          <Badge 
                            variant="secondary"
                            style={{
                              backgroundColor: colors.brand.primary,
                              color: '#ffffff'
                            }}
                          >
                            {country.value}%
                          </Badge>
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* 툴팁 */}
      <div
        id="world-tooltip"
        className="fixed pointer-events-none p-3 rounded-lg text-sm opacity-0 transition-all duration-200 z-50 border shadow-lg"
        style={{ 
          maxWidth: '200px', 
          transform: 'translateY(4px)',
          backgroundColor: isDark ? '#0b1220' : '#ffffff',
          borderColor: isDark ? '#1f2937' : '#e5e7eb',
          color: isDark ? '#e2e8f0' : '#1f2937'
        }}
      />
    </>
  );
}
