import type { Verdict } from "./api";

export const VERDICT_COLOR: Record<Verdict, string> = {
  safe:    "#16a34a",
  caution: "#eab308",
  risk:    "#dc2626",
};

export const VERDICT_LABEL: Record<Verdict, string> = {
  safe:    "SAFE",
  caution: "CAUTION",
  risk:    "RISK",
};

export const VERDICT_BG: Record<Verdict, string> = {
  safe:    "bg-green-600",
  caution: "bg-yellow-500",
  risk:    "bg-red-600",
};
