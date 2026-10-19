import { NextResponse } from "next/server";

// Static-export friendly health check (GitHub Pages build mode).
export const dynamic = "force-static";

export async function GET() {
  return NextResponse.json({ message: "WebCloth 2 API", status: "ok" });
}
