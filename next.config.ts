import type { NextConfig } from "next";

/**
 * Two build modes:
 *  - default:        standalone output for the sandbox dev/preview server
 *  - NEXT_PUBLIC_PAGES=1: static export for GitHub Pages
 *    (basePath /<repo>, trailing slashes, no server features)
 */
const isPages = process.env.NEXT_PUBLIC_PAGES === "1";
const repoName = process.env.NEXT_PUBLIC_REPO_NAME || "webcloth2";

const pagesConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  basePath: `/${repoName}`,
  assetPrefix: `/${repoName}`,
  distDir: ".next-pages",
  typescript: { ignoreBuildErrors: true },
  reactStrictMode: false,
};

const devConfig: NextConfig = {
  output: "standalone",
  typescript: { ignoreBuildErrors: true },
  reactStrictMode: false,
};

export default isPages ? pagesConfig : devConfig;

