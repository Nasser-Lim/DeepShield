from app.services.verdict import verdict_from_score


def test_high_is_risk():
    out = verdict_from_score(0.9)
    assert out.verdict == "risk"
    assert abs(out.final_score - 0.9) < 1e-6


def test_low_is_safe():
    out = verdict_from_score(0.1)
    assert out.verdict == "safe"
    assert abs(out.final_score - 0.1) < 1e-6


def test_mid_is_caution():
    out = verdict_from_score(0.5)
    assert out.verdict == "caution"


def test_boundary_safe_exclusive():
    out = verdict_from_score(0.30)
    assert out.verdict == "caution"


def test_boundary_risk_inclusive():
    out = verdict_from_score(0.70)
    assert out.verdict == "risk"
