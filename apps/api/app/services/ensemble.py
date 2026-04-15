from __future__ import annotations

from dataclasses import dataclass

from ..config import get_settings
from ..schemas.analysis import Verdict


@dataclass(frozen=True)
class Aggregated:
    final_score: float
    verdict: Verdict


def aggregate(effort: float, xray: float, spsl: float) -> Aggregated:
    """Weighted soft-vote across the three detectors."""
    s = get_settings()
    total_w = s.weight_effort + s.weight_xray + s.weight_spsl
    if total_w <= 0:
        raise ValueError("ensemble weights must sum > 0")

    final = (
        s.weight_effort * effort
        + s.weight_xray * xray
        + s.weight_spsl * spsl
    ) / total_w

    if final < s.threshold_safe:
        verdict: Verdict = "safe"
    elif final < s.threshold_risk:
        verdict = "caution"
    else:
        verdict = "risk"

    return Aggregated(final_score=round(float(final), 4), verdict=verdict)
