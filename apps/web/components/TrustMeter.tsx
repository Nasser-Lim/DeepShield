import { VERDICT_COLOR, VERDICT_LABEL } from "@/lib/verdict";
import type { Verdict } from "@/lib/api";

export function TrustMeter({ score, verdict }: { score: number; verdict: Verdict }) {
  const pct = Math.round(score * 100);
  const color = VERDICT_COLOR[verdict];
  const radius = 70;
  const circ = 2 * Math.PI * radius;
  const dash = circ * score;
  const label = `조작 확률 ${pct}% — 판정: ${VERDICT_LABEL[verdict]}`;

  return (
    <div className="flex flex-col items-center gap-2">
      <svg
        width="180" height="180" viewBox="0 0 180 180"
        role="img"
        aria-label={label}
      >
        <title>{label}</title>
        {/* Track */}
        <circle cx="90" cy="90" r={radius} fill="none" stroke="#e5e7eb" strokeWidth="14" />
        {/* Progress */}
        <circle
          cx="90" cy="90" r={radius}
          fill="none"
          stroke={color}
          strokeWidth="14"
          strokeDasharray={`${dash} ${circ - dash}`}
          strokeDashoffset={circ / 4}
          strokeLinecap="round"
          transform="rotate(-90 90 90)"
          style={{ transition: "stroke-dasharray 0.6s cubic-bezier(0.25,1,0.5,1), stroke 0.3s" }}
        />
        {/* Score text */}
        <text x="90" y="86" textAnchor="middle" fontSize="30" fontWeight="700" fill="#0f172a">
          {pct}%
        </text>
        <text x="90" y="108" textAnchor="middle" fontSize="11" fill="#64748b" letterSpacing="0.02em">
          조작 확률
        </text>
      </svg>
    </div>
  );
}
