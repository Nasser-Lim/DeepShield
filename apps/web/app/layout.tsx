import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DeepShield — Newsroom Trust Defense",
  description: "Triple-voting deepfake detection for newsrooms.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body>
        <header className="border-b border-slate-200 bg-white">
          <div className="mx-auto max-w-6xl px-6 py-4 flex items-center gap-3">
            {/* Shield icon */}
            <svg
              width="32" height="32" viewBox="0 0 32 32" fill="none"
              aria-hidden="true"
              className="shrink-0"
            >
              <rect width="32" height="32" rx="8" fill="#0f172a" />
              <path
                d="M16 6L8 9.5V16c0 4.2 3.3 7.8 8 9 4.7-1.2 8-4.8 8-9V9.5L16 6z"
                fill="white" fillOpacity="0.15" stroke="white" strokeWidth="1.5"
                strokeLinejoin="round"
              />
              <path
                d="M12.5 16.5l2.5 2.5 4.5-4.5"
                stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"
              />
            </svg>
            <div>
              <div className="text-lg font-semibold tracking-tight">DeepShield</div>
              <div className="text-xs text-slate-500">뉴스룸 신뢰도 방어 — 삼중 투표제</div>
            </div>
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
