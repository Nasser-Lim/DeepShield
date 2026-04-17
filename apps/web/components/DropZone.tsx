"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { analyzeImage, type AnalyzeResult } from "@/lib/api";

const STEPS = ["이미지 업로드 중…", "탐지기 추론 중…", "결과 집계 중…"];

function Spinner() {
  return (
    <svg
      className="animate-spin h-6 w-6 text-slate-400"
      xmlns="http://www.w3.org/2000/svg"
      fill="none" viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}

export function DropZone({ onResult }: { onResult: (r: AnalyzeResult, file: File) => void }) {
  const [busy, setBusy] = useState(false);
  const [step, setStep] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(
    async (files: File[]) => {
      const file = files[0];
      if (!file) return;
      setBusy(true);
      setError(null);
      setStep(0);

      // Fake step progression while waiting for the real response
      const timer1 = setTimeout(() => setStep(1), 600);
      const timer2 = setTimeout(() => setStep(2), 2000);

      try {
        const result = await analyzeImage(file);
        onResult(result, file);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        clearTimeout(timer1);
        clearTimeout(timer2);
        setBusy(false);
        setStep(0);
      }
    },
    [onResult],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [] },
    multiple: false,
    disabled: busy,
  });

  return (
    <div
      {...getRootProps()}
      className={[
        "cursor-pointer rounded-xl border-2 border-dashed p-6 sm:p-10 text-center",
        "transition-colors duration-200",
        isDragActive
          ? "border-slate-900 bg-slate-100"
          : "border-slate-300 bg-white hover:bg-slate-50 hover:border-slate-400",
        busy ? "pointer-events-none opacity-70" : "",
      ].join(" ")}
      aria-busy={busy}
      aria-label="이미지 업로드 영역"
    >
      <input {...getInputProps()} />

      {busy ? (
        <div className="flex flex-col items-center gap-3">
          <Spinner />
          <span className="text-sm font-medium text-slate-600">{STEPS[step]}</span>
        </div>
      ) : (
        <>
          <div className="text-base sm:text-lg font-medium text-slate-800">
            {isDragActive ? "여기에 놓으세요" : "제보 이미지를 드래그하거나 클릭해 업로드"}
          </div>
          <div className="mt-1 text-sm text-slate-500">
            DIRE 모델이 전체 이미지를 재구성하여 디퓨전 생성 여부를 판정합니다
          </div>
        </>
      )}

      {error && (
        <div role="alert" className="mt-3 text-sm text-red-600 bg-red-50 rounded-md px-3 py-2">
          {error}
        </div>
      )}
    </div>
  );
}
