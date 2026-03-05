import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/data/:path*",
        destination: `${process.env.NEXT_PUBLIC_DATA_API_URL || "http://localhost:8000"}/api/:path*`,
      },
      {
        source: "/api/chatbot/:path*",
        destination: `${process.env.NEXT_PUBLIC_CHATBOT_API_URL || "http://localhost:8001"}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
