"use client";

import type { InferResponse } from "@/lib/api";

export function ModelScoreTabs({ scores }: { scores: InferResponse }) {
  const current = scores.dire;
  const pct = Math.round(current.score * 100);
  const tone =
    pct >= 70 ? "text-red-500" : pct >= 30 ? "text-yellow-600" : "text-green-600";
  const bar =
    pct >= 70 ? "bg-red-500" : pct >= 30 ? "bg-yellow-400" : "bg-green-500";

  return (
    <div className="rounded-lg border border-slate-200 bg-white overflow-hidden">
      <div className="flex items-center justify-between border-b border-slate-200 bg-slate-900 px-4 py-3 text-white">
        <div className="flex items-center gap-2">
          <span className="font-semibold">DIRE</span>
          <span className="text-xs font-mono bg-slate-700 text-slate-200 rounded px-1.5 py-0.5">
            단일 모델
          </span>
        </div>
        <span className={`font-mono font-bold ${pct >= 70 ? "text-red-300" : pct >= 30 ? "text-yellow-200" : "text-green-300"}`}>
          {pct}%
        </span>
      </div>

      <div className="p-4 space-y-3">
        <div className="flex items-center gap-3">
          <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${bar}`}
              style={{ width: `${pct}%` }}
            />
          </div>
          <span className={`font-mono text-sm font-bold w-10 text-right ${tone}`}>
            {pct}%
          </span>
        </div>

        <div className="rounded-md bg-slate-50 border border-slate-100 px-3 py-2.5">
          <div className="text-xs font-semibold text-slate-700 mb-1">
            Diffusion Reconstruction Error
          </div>
          <p className="text-xs text-slate-500 leading-relaxed">
            ADM(Ablated Diffusion Model)으로 이미지를 DDIM 왕복 재구성한 뒤 원본과의
            픽셀 차이 맵(DIRE)을 ResNet-50 이진 분류기에 입력합니다. Midjourney · Stable
            Diffusion · Nano-Banana 등 디퓨전 계열 생성 이미지 전반에 반응하도록 설계된
            전체-이미지 탐지기입니다. (ICCV 2023, Wang et al.)
          </p>
        </div>

        {current.heatmap_b64 ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={`data:image/png;base64,${current.heatmap_b64}`}
            alt="DIRE 재구성 오차 히트맵"
            aria-hidden="true"
            className="w-full max-w-md rounded-md border border-slate-200"
          />
        ) : (
          <p className="text-xs text-slate-400">히트맵 없음</p>
        )}
      </div>
    </div>
  );
}
