/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  outputFileTracingRoot: __dirname,
  // TanStack Table v9 ships ESM only. Listing it here also makes next/jest transform it
  // (Jest runs CommonJS); the production build bundles ESM either way.
  transpilePackages: ["@tanstack/react-table", "@tanstack/table-core"],
};

module.exports = nextConfig;
