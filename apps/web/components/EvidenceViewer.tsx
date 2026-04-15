"use client";

import { useState } from "react";

export function EvidenceViewer({
  originalUrl,
  overlayB64,
}: {
  originalUrl: string;
  overlayB64: string;
}) {
  const [alpha, setAlpha] = useState(0.6);

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="mb-3 flex items-center justify-between gap-4">
        <h3 className="font-semibold text-slate-800 shrink-0">히트맵 오버레이</h3>
        <div className="flex items-center gap-2 text-sm text-slate-600 min-w-0">
          <label htmlFor="opacity-slider" className="shrink-0">투명도</label>
          <input
            id="opacity-slider"
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={alpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
            className="w-24 sm:w-32 accent-slate-900"
            aria-valuetext={`${Math.round(alpha * 100)}%`}
          />
          <span className="tabular-nums w-9 text-right text-slate-500 shrink-0">
            {Math.round(alpha * 100)}%
          </span>
        </div>
      </div>

      <div className="relative mx-auto max-w-xl">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={originalUrl}
          alt="분석 대상 원본 이미지"
          className="w-full rounded-md block"
        />
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`data:image/png;base64,${overlayB64}`}
          alt=""
          aria-hidden="true"
          className="absolute inset-0 w-full h-full rounded-md pointer-events-none mix-blend-multiply"
          style={{ opacity: alpha, transition: "opacity 0.15s" }}
        />
      </div>
    </div>
  );
}
