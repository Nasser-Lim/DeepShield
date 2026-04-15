"use client";

import { useState } from "react";
import type { InferResponse, ModelScore } from "@/lib/api";

type TabKey = "effort" | "xray" | "spsl";

const TABS: { key: TabKey; label: string; blurb: string }[] = [
  { key: "effort", label: "Xception",  blurb: "생성 모델 고유 지문 분석" },
  { key: "xray",   label: "F3Net",     blurb: "이미지 합성 경계선 분석" },
  { key: "spsl",   label: "SPSL",      blurb: "주파수 위상 스펙트럼 분석" },
];

export function ModelScoreTabs({ scores }: { scores: InferResponse }) {
  const [active, setActive] = useState<TabKey>("effort");
  const current: ModelScore = scores[active];
  const pct = Math.round(current.score * 100);

  return (
    <div className="rounded-lg border border-slate-200 bg-white overflow-hidden">
      {/* Tab list */}
      <div role="tablist" aria-label="탐지기별 점수" className="flex border-b border-slate-200">
        {TABS.map((t) => {
          const isActive = active === t.key;
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
              <div>{t.label}</div>
              <div className={`text-xs mt-0.5 ${isActive ? "text-slate-300" : "text-slate-400"}`}>
                {Math.round(scores[t.key].score * 100)}%
              </div>
            </button>
          );
        })}
      </div>

      {/* Tab panel */}
      <div
        role="tabpanel"
        id={`tabpanel-${active}`}
        aria-labelledby={`tab-${active}`}
        className="p-4"
      >
        <p className="mb-3 text-sm text-slate-500">
          {TABS.find((t) => t.key === active)?.blurb}
          {" — "}
          <span className="font-mono font-medium text-slate-700">{pct}%</span>
        </p>
        {current.heatmap_b64 ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={`data:image/png;base64,${current.heatmap_b64}`}
            alt={`${TABS.find((t) => t.key === active)?.label} 히트맵`}
            className="w-full max-w-md rounded-md border border-slate-200"
          />
        ) : (
          <p className="text-sm text-slate-400">히트맵 없음</p>
        )}
      </div>
    </div>
  );
}
