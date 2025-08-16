"use client";
import React, { useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface CountryData {
  name: string;
  value: number;
  rank: number;
}

interface WorldInfidelityMapProps {
  className?: string;
}

export function WorldInfidelityMap({ className }: WorldInfidelityMapProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [data, setData] = useState<Map<string, CountryData>>(new Map());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

  useEffect(() => {
    setData(new Map(Object.entries(countryData)));
    setLoading(false);
  }, []);

  useEffect(() => {
    if (!data.size || !svgRef.current) return;

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
  }, [data]);

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
        .fitExtent([[12, 12], [width - 12, height - 12]], { type: 'Sphere' });
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
        .attr('fill', '#0b111c');

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
          return item ? color(item.value) : '#cfd4dc';
        })
        .attr('stroke', 'rgba(0,0,0,.35)')
        .attr('stroke-width', 0.5)
        .style('cursor', 'pointer')
        .on('mousemove', function(this: any, event: any, d: any) {
          const [x, y] = d3.pointer(event);
          const code = d.id ? ISO3_from_numeric(d.id) : null;
          const item: CountryData | undefined  = code ? data.get(code) : undefined;
          const name = item?.name || d.properties.name || 'Not ranked';

          d3.select(this)
            .attr('stroke', 'white')
            .attr('stroke-width', 1.2)
            .attr('opacity', 0.95);

          if (item) {
            showTooltip(event, name, item);
          }
        })
        .on('mouseleave', function(this: any) {
          d3.select(this)
            .attr('stroke', 'rgba(0,0,0,.35)')
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

  if (loading) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <p className="text-gray-600">세계 지도 데이터를 불러오는 중...</p>
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

  // 상위 10개 국가 정렬
  const topCountries = Array.from(data.entries())
    .map(([iso3, obj]) => ({ iso3, ...obj }))
    .sort((a, b) => a.rank - b.rank)
    .slice(0, 10);

  return (
    <>
      <Card className={className}>
        <CardHeader>
          <CardTitle className="text-2xl font-bold">🌍 세계 국가별 바람지수 2025</CardTitle>
          <p className="text-gray-600">
            나라별 바람 지수(자기보고형 설문 %). 파란색이 진할수록 비율이 높습니다. 순위에 들지 않은 국가는 회색으로 표시됩니다.
          </p>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {/* 범례 */}
            <div className="flex items-center gap-2 text-sm text-gray-600 flex-wrap">
              <span className="font-semibold text-gray-900">{"Cheated(%)"}</span>
              {[10, 15, 20, 25, 30, 35, 40, 45, 50, 55].map((value) => {
                const color = value === 10 ? '#c7f9cc' : 
                             value === 55 ? '#0b3c8a' : 
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
            </div>

            {/* 지도 */}
            <div className="bg-gray-50 rounded-lg p-4 h-[600px] w-full">
              <svg 
                ref={svgRef}
                className="w-full h-full"
                role="img" 
                aria-label="World choropleth map of infidelity rates"
                style={{ minHeight: '600px' }}
              />
            </div>

            {/* 순위표 */}
            <div className="mt-6">
              <h3 className="text-lg font-semibold mb-3">상위 국가 순위</h3>
              <div className="bg-gray-50 rounded-lg overflow-hidden">
                <table className="w-full">
                  <thead className="bg-gray-100">
                    <tr>
                      <th className="text-left p-3 font-semibold text-gray-700 w-[70%]">국가</th>
                      <th className="text-left p-3 font-semibold text-gray-700 w-[30%]">바람지수</th>
                    </tr>
                  </thead>
                  <tbody>
                    {topCountries.map((country, index) => (
                      <tr key={country.iso3} className="border-t border-gray-200 hover:bg-gray-50">
                        <td className="p-3">
                          <div className="flex items-center gap-3">
                            {ISO3_TO_A2[country.iso3] && (
                              <img 
                                className="w-5 h-4 rounded border border-gray-300"
                                src={`https://flagcdn.com/w20/${ISO3_TO_A2[country.iso3]}.png`}
                                alt={`${country.name} flag`}
                                loading="lazy"
                              />
                            )}
                            <span>{country.name}</span>
                          </div>
                        </td>
                        <td className="p-3">
                          <Badge variant="secondary">
                            {Math.round(country.value)}%
                          </Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="p-3 text-xs text-gray-500 bg-gray-100">
                  참고: 표는 <strong>Rank</strong> 오름차순으로 정렬되며, <strong>국가</strong>와 <strong>바람지수</strong>만 표시합니다. (국기 아이콘 포함)
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 툴팁 */}
      <div
        id="world-tooltip"
        className="fixed pointer-events-none bg-gray-900 text-white p-3 rounded-lg text-sm opacity-0 transition-all duration-200 z-50 border border-gray-700 shadow-lg"
        style={{ maxWidth: '200px', transform: 'translateY(4px)' }}
      />
    </>
  );
}
