"use client";

import { useEffect, useRef, useState } from "react";
import { DropZone } from "@/components/DropZone";
import { TrustMeter } from "@/components/TrustMeter";
import { EvidenceViewer } from "@/components/EvidenceViewer";
import { ModelScoreTabs } from "@/components/ModelScoreTabs";
import { ReportExport } from "@/components/ReportExport";
import { VERDICT_COLOR, VERDICT_LABEL, VERDICT_BG } from "@/lib/verdict";
import type { AnalyzeResult } from "@/lib/api";

export default function Home() {
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const prevUrl = useRef<string | null>(null);

  // Revoke previous object URL to avoid memory leaks
  useEffect(() => {
    return () => {
      if (prevUrl.current) URL.revokeObjectURL(prevUrl.current);
    };
  }, []);

  const handleResult = (r: AnalyzeResult, file: File) => {
    if (prevUrl.current) URL.revokeObjectURL(prevUrl.current);
    const url = URL.createObjectURL(file);
    prevUrl.current = url;
    setOriginalUrl(url);
    setResult(r);
  };

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-2xl font-bold mb-2 text-slate-900">이미지 진위 검증</h1>
        <p className="text-slate-600 mb-4 leading-relaxed">
          제보 이미지를 드래그하거나 클릭해 업로드하세요.{" "}
          <strong className="font-semibold text-slate-800">Effort · Face X-ray · SPSL</strong>{" "}
          세 탐지기가 동시에 추론하고 가중 투표합니다.
        </p>
        <DropZone onResult={handleResult} />
      </section>

      {result && (
        <section
          className="grid grid-cols-1 lg:grid-cols-3 gap-6"
          aria-label="분석 결과"
        >
          {/* Verdict panel */}
          <div className="lg:col-span-1 rounded-lg border border-slate-200 bg-white p-6 flex flex-col items-center gap-4">
            <TrustMeter score={result.final_score} verdict={result.verdict} />

            {/* Single verdict badge */}
            <div
              className={`w-full text-center px-3 py-2 rounded-lg text-sm font-bold text-white tracking-wide ${VERDICT_BG[result.verdict]}`}
              style={{ backgroundColor: VERDICT_COLOR[result.verdict] }}
            >
              {VERDICT_LABEL[result.verdict]}
            </div>

            <div className="text-xs text-slate-400 font-mono break-all text-center">
              {result.id}
            </div>
            <ReportExport result={result} />
          </div>

          {/* Evidence panel */}
          <div className="lg:col-span-2 space-y-6">
            {originalUrl && (
              <EvidenceViewer
                originalUrl={originalUrl}
                overlayB64={result.scores.overlay_b64}
              />
            )}
            <ModelScoreTabs scores={result.scores} />
          </div>
        </section>
      )}
    </div>
  );
}
