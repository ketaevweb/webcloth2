import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "WebCloth 2 — производительный аналог Magica Cloth 2 для WebGPU/WebGL 2+",
  description:
    "Открытая физика ткани в браузере: XPBD-движок с compute-шейдерами WebGPU и CPU-фолбэком для WebGL 2+. Демо, исходный код, сравнение с Magica Cloth 2.",
  keywords: ["WebGPU", "WebGL2", "cloth simulation", "физика ткани", "XPBD", "Magica Cloth", "three.js"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ru" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
