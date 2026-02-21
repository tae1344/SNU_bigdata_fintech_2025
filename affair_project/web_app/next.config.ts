import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  eslint: {
    // build 시 ESLint 검사
    ignoreDuringBuilds: false,
  },
  typescript: {
    // build 시 TypeScript 검사
    ignoreBuildErrors: false,
  },
  
  // 빌드 최적화
  swcMinify: true,
  
  // 이미지 최적화
  images: {
    unoptimized: false,
  },
  
  // 정적 사이트 생성 (SSG) 지원
  trailingSlash: true,
  
  // 환경 변수 설정
  env: {
    CUSTOM_KEY: process.env.CUSTOM_KEY,
  },
  
  // API 라우트 최적화
  experimental: {
    // 서버 컴포넌트 최적화
    serverComponentsExternalPackages: [],
  },
};

export default nextConfig;
