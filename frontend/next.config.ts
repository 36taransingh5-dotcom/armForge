import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Without this, Next.js infers the workspace root by walking up for
  // lockfiles and can land on an unrelated one elsewhere on disk.
  outputFileTracingRoot: __dirname,
};

export default nextConfig;
