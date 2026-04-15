"use client";

import type { AnalyzeResult } from "@/lib/api";

export function ReportExport({ result }: { result: AnalyzeResult }) {
  const onExport = () => {
    // Minimal print-based PDF export: opens a printable tab.
    // Swap for @react-pdf/renderer when a formal template is defined.
    const w = window.open("", "_blank", "width=800,height=1000");
    if (!w) return;
    const score = result.final_score ?? 0;
    const verdict = result.verdict ?? "unknown";
    const s = result.scores;
    w.document.write(`<!doctype html><html><head><title>DeepShield Report ${result.id}</title>
<style>
  body { font-family: system-ui, sans-serif; padding: 32px; color: #0f172a; }
  h1 { margin: 0 0 8px; }
  .verdict { display:inline-block; padding: 4px 10px; border-radius: 999px; color: white;
             background: ${verdict === "safe" ? "#16a34a" : verdict === "caution" ? "#eab308" : "#dc2626"}; }
  table { border-collapse: collapse; margin-top: 16px; }
  td,th { border: 1px solid #cbd5e1; padding: 6px 12px; text-align: left; }
  img { max-width: 480px; margin-top: 16px; border: 1px solid #e2e8f0; }
</style></head><body>
<h1>DeepShield Verification Report</h1>
<div>Analysis ID: <code>${result.id}</code></div>
<div style="margin-top:8px;">Verdict: <span class="verdict">${verdict.toUpperCase()}</span>
  &nbsp; Final score: <b>${(score * 100).toFixed(1)}%</b></div>
<table>
  <tr><th>Model</th><th>Probability</th></tr>
  <tr><td>Effort</td><td>${s ? (s.effort.score * 100).toFixed(1) : "-"}%</td></tr>
  <tr><td>Face X-ray</td><td>${s ? (s.xray.score * 100).toFixed(1) : "-"}%</td></tr>
  <tr><td>SPSL</td><td>${s ? (s.spsl.score * 100).toFixed(1) : "-"}%</td></tr>
</table>
${s ? `<img src="data:image/png;base64,${s.overlay_b64}" alt="overlay" />` : ""}
<script>window.onload = () => window.print();</script>
</body></html>`);
    w.document.close();
  };

  return (
    <button
      onClick={onExport}
      className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700"
    >
      Export PDF Report
    </button>
  );
}
