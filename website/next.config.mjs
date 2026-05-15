/** @type {import('next').NextConfig} */

// GitHub Pages serves project sites under /<repo>. The Pages workflow sets
// PAGES=true so basePath only kicks in during the deploy build; local
// `next dev` keeps serving at root.
const isPages = process.env.PAGES === "true";
const REPO = "PlantCLEF2026";
const basePath = isPages ? `/${REPO}` : "";

const nextConfig = {
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,

  basePath:    basePath || undefined,
  assetPrefix: basePath ? basePath + "/" : undefined,

  // Expose basePath to client code so plain <img src> can be prefixed via
  // lib/paths.asset(). Next.js does not auto-rewrite raw <img> srcs.
  env: {
    NEXT_PUBLIC_BASE_PATH: basePath,
  },

  reactStrictMode: true,
};

export default nextConfig;
