"use client";

import { useState } from "react";
import type { InferResponse, ModelScore } from "@/lib/api";

type TabKey = "effort" | "xray" | "spsl";

const TABS: { key: TabKey; label: string; weight: string; blurb: string; detail: string }[] = [
  {
    key: "effort",
    label: "Xception",
    weight: "40%",
    blurb: "CNN 텍스처 아티팩트 탐지",
    detail: "딥러닝 생성 이미지에서 나타나는 픽셀 수준의 텍스처 불일치를 Xception 아키텍처로 분류합니다. GAN·Diffusion 모델의 고주파 아티팩트에 민감합니다.",
  },
  {
    key: "xray",
    label: "F3Net",
    weight: "35%",
    blurb: "주파수 분해 합성 경계 탐지",
    detail: "FAD(주파수 인식 분해) 헤드가 DCT 4개 대역으로 이미지를 분해한 뒤 Xception이 각 대역의 이상 패턴을 학습합니다. 얼굴 합성 경계선의 주파수 불연속성에 강합니다.",
  },
  {
    key: "spsl",
    label: "SPSL",
    weight: "25%",
    blurb: "위상 스펙트럼 얕은 학습",
    detail: "RGB에 FFT 위상 스펙트럼 채널을 추가한 4채널 입력으로 Xception을 훈련합니다. 생성 모델이 공간 도메인에서 숨기기 어려운 위상 불일치를 직접 포착합니다.",
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
