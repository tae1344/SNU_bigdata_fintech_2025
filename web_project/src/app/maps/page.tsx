"use client";
import { InfidelityMap } from "@/components/legacy/InfidelityMap";
import { WorldInfidelityMap } from "@/components/legacy/WorldInfidelityMap";

export default function MapPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
     <section className="py-16 bg-white">
      <div className="container mx-auto px-4">
        {/* 지도 그리드 */}
        <div className="flex flex-col gap-8 mb-8">
          {/* 세계 지도 */}
          <WorldInfidelityMap />

          {/* 미국 지도 */}
          <InfidelityMap />
        </div>
      </div>
    </section>
    </div>
  );
}
