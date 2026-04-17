from __future__ import annotations

from dataclasses import dataclass

from ..config import get_settings
from ..schemas.analysis import Verdict


@dataclass(frozen=True)
class Aggregated:
    final_score: float
    verdict: Verdict


def verdict_from_score(score: float) -> Aggregated:
    """Map the single DIRE synthetic-probability into safe/caution/risk."""
    s = get_settings()
    if score < s.threshold_safe:
        v: Verdict = "safe"
    elif score < s.threshold_risk:
        v = "caution"
    else:
        v = "risk"
    return Aggregated(final_score=round(float(score), 4), verdict=v)
