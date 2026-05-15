/** @type {import('next').NextConfig} */

// GitHub Pages serves project sites under /<repo>. The Pages workflow sets
// PAGES=true so basePath only kicks in during the deploy build; local
// `next dev` keeps serving at root.
const isPages = process.env.PAGES === "true";
const REPO = "PlantCLEF2026";

const nextConfig = {
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,

  basePath:    isPages ? `/${REPO}`  : undefined,
  assetPrefix: isPages ? `/${REPO}/` : undefined,

  reactStrictMode: true,
};

export default nextConfig;
