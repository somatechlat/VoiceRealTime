/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: "standalone",
  env: {
    // Portal API (port 25001)
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:25001",
    // Gateway API (port 25000) - Voice sessions, workers
    NEXT_PUBLIC_GATEWAY_URL: process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:25000",
    // Keycloak (port 25004)
    NEXT_PUBLIC_KEYCLOAK_URL: process.env.NEXT_PUBLIC_KEYCLOAK_URL || "http://localhost:25004",
    NEXT_PUBLIC_KEYCLOAK_REALM: process.env.NEXT_PUBLIC_KEYCLOAK_REALM || "agentvoicebox",
    NEXT_PUBLIC_KEYCLOAK_CLIENT_ID: process.env.NEXT_PUBLIC_KEYCLOAK_CLIENT_ID || "portal-frontend",
    // Lago Billing (port 25005)
    NEXT_PUBLIC_LAGO_URL: process.env.NEXT_PUBLIC_LAGO_URL || "http://localhost:25005",
    // Prometheus (port 25008)
    NEXT_PUBLIC_PROMETHEUS_URL: process.env.NEXT_PUBLIC_PROMETHEUS_URL || "http://localhost:25008",
    // Grafana (port 25009)
    NEXT_PUBLIC_GRAFANA_URL: process.env.NEXT_PUBLIC_GRAFANA_URL || "http://localhost:25009",
  },
  async rewrites() {
    return [
      {
        source: "/api/v1/:path*",
        destination: `${process.env.PORTAL_API_URL || "http://localhost:25001"}/v1/:path*`,
      },
      {
        source: "/gateway/:path*",
        destination: `${process.env.GATEWAY_URL || "http://localhost:25000"}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
