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
};

export default nextConfig;
