"use client";

import { useState } from "react";
import type { InferResponse, ModelScore } from "@/lib/api";

type TabKey = "effort" | "xray" | "spsl";

const TABS: { key: TabKey; label: string; weight: string; blurb: string; detail: string }[] = [
  {
    key: "effort",
    label: "SBI",
    weight: "40%",
    blurb: "자가 혼합 이미지 합성 흔적 탐지",
    detail: "EfficientNet-B4 기반. 얼굴 합성 시 발생하는 블렌딩 경계 아티팩트를 학습합니다. GAN·Diffusion 모두에서 나타나는 합성 흔적에 민감하며 실사 사진 오탐률이 낮습니다. (CVPR 2022)",
  },
  {
    key: "xray",
    label: "FatFormer",
    weight: "40%",
    blurb: "CLIP 기반 위조 적응형 트랜스포머",
    detail: "CLIP ViT-L/14 백본에 위조 인식 어댑터를 추가했습니다. 주파수 도메인 분석과 언어-이미지 정렬을 결합해 학습하지 않은 생성 모델에도 높은 일반화 성능을 보입니다. (CVPR 2024)",
  },
  {
    key: "spsl",
    label: "C2P-CLIP",
    weight: "20%",
    blurb: "카테고리 공통 프롬프트 CLIP 탐지",
    detail: "CLIP ViT-L/14에 카테고리 공통 프롬프트(C2P)를 주입해 실제·딥페이크 텍스트 임베딩과의 유사도로 판별합니다. 실사 사진에 대한 오탐률이 매우 낮습니다. (AAAI 2025)",
  },
];

export function ModelScoreTabs({ scores }: { scores: InferResponse }) {
  const [active, setActive] = useState<TabKey>("effort");
  const current: ModelScore = scores[active];
  const pct = Math.round(current.score * 100);

  return (
    <div className="rounded-lg border border-slate-200 bg-white overflow-hidden">
      {/* Tab list */}
      <div role="tablist" aria-label="모델별 점수" className="flex border-b border-slate-200">
        {TABS.map((t) => {
          const isActive = active === t.key;
          const scorePct = Math.round(scores[t.key].score * 100);
          return (
            <button
              key={t.key}
              role="tab"
              aria-selected={isActive}
              aria-controls={`tabpanel-${t.key}`}
              id={`tab-${t.key}`}
              onClick={() => setActive(t.key)}
              className={[
                "flex-1 px-4 py-3 text-sm font-medium transition-colors duration-150",
                isActive
                  ? "bg-slate-900 text-white"
                  : "text-slate-600 hover:bg-slate-50 hover:text-slate-900",
              ].join(" ")}
            >
              <div className="flex items-center justify-center gap-1.5">
                <span>{t.label}</span>
                <span className={`text-xs px-1.5 py-0.5 rounded font-mono ${isActive ? "bg-slate-700 text-slate-300" : "bg-slate-100 text-slate-500"}`}>
                  {t.weight}
                </span>
              </div>
              <div className={`text-xs mt-0.5 font-mono font-semibold ${isActive ? "text-white" : scorePct >= 70 ? "text-red-500" : scorePct >= 30 ? "text-yellow-600" : "text-green-600"}`}>
                {scorePct}%
              </div>
            </button>
          );
        })}
      </div>

      {/* Tab panel */}
      {TABS.map((t) => {
        const isActive = active === t.key;
        const tabScore = scores[t.key];
        const tabPct = Math.round(tabScore.score * 100);
        return (
          <div
            key={t.key}
            role="tabpanel"
            id={`tabpanel-${t.key}`}
            aria-labelledby={`tab-${t.key}`}
            hidden={!isActive}
            className="p-4 space-y-3"
          >
            {/* Score bar */}
            <div className="flex items-center gap-3">
              <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    tabPct >= 70 ? "bg-red-500" : tabPct >= 30 ? "bg-yellow-400" : "bg-green-500"
                  }`}
                  style={{ width: `${tabPct}%` }}
                />
              </div>
              <span className="font-mono text-sm font-bold text-slate-800 w-10 text-right">
                {tabPct}%
              </span>
            </div>

            {/* Model description */}
            <div className="rounded-md bg-slate-50 border border-slate-100 px-3 py-2.5">
              <div className="text-xs font-semibold text-slate-700 mb-1">{t.blurb}</div>
              <p className="text-xs text-slate-500 leading-relaxed">{t.detail}</p>
            </div>

            {/* Heatmap */}
            {tabScore.heatmap_b64 ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={`data:image/png;base64,${tabScore.heatmap_b64}`}
                alt={`${t.label} 히트맵`}
                aria-hidden="true"
                className="w-full max-w-md rounded-md border border-slate-200"
              />
            ) : (
              <p className="text-xs text-slate-400">히트맵 없음 (실제 모델 추론 시 생성)</p>
            )}
          </div>
        );
      })}
    </div>
  );
}
