from app.services.ensemble import aggregate


def test_all_high_is_risk():
    out = aggregate(0.9, 0.9, 0.9)
    assert out.verdict == "risk"
    assert abs(out.final_score - 0.9) < 1e-6


def test_all_low_is_safe():
    out = aggregate(0.1, 0.1, 0.1)
    assert out.verdict == "safe"
    assert abs(out.final_score - 0.1) < 1e-6


def test_mid_is_caution():
    out = aggregate(0.5, 0.5, 0.5)
    assert out.verdict == "caution"
