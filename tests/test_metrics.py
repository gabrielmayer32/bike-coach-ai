"""
Unit tests for the metric computation layer.
All deterministic — no mocks, no API calls, no DB.
"""

import pytest
from app.metrics.compute import (
    normalised_power,
    intensity_factor,
    variability_index,
    pw_hr_decoupling,
    rep_fade_pct,
    within_rep_fade_pct,
    power_compliance_pct,
    peak_power_for_duration,
    avg_cadence,
    cadence_below_threshold_pct,
    avg_hr,
    max_hr,
    hr_near_max_pct,
    time_in_zones_from_stream,
)


# ── normalised_power ──────────────────────────────────────────────────────────

def test_np_steady_power():
    # Constant power → NP should equal that power
    watts = [250.0] * 300
    result = normalised_power(watts)
    assert result == pytest.approx(250.0, abs=1.0)


def test_np_with_spikes():
    # Spiky power raises NP above avg (4th-power weighting)
    base = [200.0] * 240
    spikes = [400.0] * 60
    watts = base + spikes
    avg = (200 * 240 + 400 * 60) / 300   # 260
    result = normalised_power(watts)
    assert result is not None
    assert result > avg


def test_np_too_short():
    assert normalised_power([200.0] * 10) is None


def test_np_all_none():
    assert normalised_power([None] * 100) is None


# ── intensity_factor ──────────────────────────────────────────────────────────

def test_if_at_ftp():
    assert intensity_factor(280, 280) == pytest.approx(1.0, abs=0.001)


def test_if_below_ftp():
    result = intensity_factor(210, 280)
    assert result == pytest.approx(0.75, abs=0.001)


def test_if_none_ftp():
    assert intensity_factor(280, 0) is None


# ── variability_index ─────────────────────────────────────────────────────────

def test_vi_steady():
    assert variability_index(200.0, 200.0) == pytest.approx(1.0)


def test_vi_spiky():
    result = variability_index(220.0, 190.0)
    assert result == pytest.approx(220 / 190, abs=0.001)


def test_vi_zero_avg():
    assert variability_index(220.0, 0.0) is None


# ── decoupling ────────────────────────────────────────────────────────────────

def test_decoupling_no_drift():
    # Constant power and HR → near-zero decoupling
    power = [200.0] * 200
    hr = [140.0] * 200
    result = pw_hr_decoupling(power, hr)
    assert result == pytest.approx(0.0, abs=0.5)


def test_decoupling_positive_drift():
    # HR rises in second half at same power → EF falls → decoupling is negative (ef2 < ef1)
    power = [200.0] * 200
    hr = [130.0] * 100 + [150.0] * 100   # HR drifts up in second half
    result = pw_hr_decoupling(power, hr)
    assert result is not None
    assert result > 0   # EF degraded: HR drifted up → decoupling positive


def test_decoupling_too_short():
    assert pw_hr_decoupling([200.0] * 30, [140.0] * 30) is None


def test_decoupling_no_hr():
    assert pw_hr_decoupling([200.0] * 200, []) is None


# ── rep_fade_pct ──────────────────────────────────────────────────────────────

def test_rep_fade_no_fade():
    assert rep_fade_pct([280.0, 280.0, 280.0]) == pytest.approx(0.0)


def test_rep_fade_positive():
    # 300 → 270 = 10% fade
    result = rep_fade_pct([300.0, 285.0, 270.0])
    assert result == pytest.approx(10.0, abs=0.1)


def test_rep_fade_negative_split():
    # Building — negative value (good sign)
    result = rep_fade_pct([260.0, 270.0, 280.0])
    assert result < 0


def test_rep_fade_single_rep():
    assert rep_fade_pct([280.0]) is None


def test_rep_fade_with_nones():
    result = rep_fade_pct([300.0, None, 270.0])
    assert result == pytest.approx(10.0, abs=0.1)


# ── within_rep_fade_pct ───────────────────────────────────────────────────────

def test_within_rep_even():
    power = [280.0] * 60
    assert within_rep_fade_pct(power) == pytest.approx(0.0, abs=0.5)


def test_within_rep_fade():
    # First half 300W, second half 260W → ~13% fade
    power = [300.0] * 30 + [260.0] * 30
    result = within_rep_fade_pct(power)
    assert result is not None
    assert result > 10


def test_within_rep_too_short():
    assert within_rep_fade_pct([280.0] * 10) is None


# ── power_compliance_pct ──────────────────────────────────────────────────────

def test_compliance_exact():
    assert power_compliance_pct(280.0, 280.0) == pytest.approx(0.0)


def test_compliance_over():
    result = power_compliance_pct(294.0, 280.0)
    assert result == pytest.approx(5.0, abs=0.1)


def test_compliance_under():
    result = power_compliance_pct(266.0, 280.0)
    assert result == pytest.approx(-5.0, abs=0.1)


# ── peak_power_for_duration ───────────────────────────────────────────────────

def test_peak_power_5s():
    watts = [200.0] * 100 + [500.0] * 10 + [200.0] * 100
    result = peak_power_for_duration(watts, 5)
    assert result == pytest.approx(500.0, abs=1.0)


def test_peak_power_too_short():
    assert peak_power_for_duration([300.0] * 3, 5) is None


def test_peak_power_with_nones():
    watts = [None] * 50 + [400.0] * 20 + [200.0] * 50
    result = peak_power_for_duration(watts, 5)
    assert result is not None
    assert result == pytest.approx(400.0, abs=1.0)


# ── cadence ───────────────────────────────────────────────────────────────────

def test_avg_cadence_basic():
    cad = [85.0, 87.0, 83.0, 85.0]
    assert avg_cadence(cad) == pytest.approx(85.0, abs=0.5)


def test_avg_cadence_skip_zeros():
    cad = [0.0, 85.0, 85.0, 0.0]
    assert avg_cadence(cad) == pytest.approx(85.0)


def test_avg_cadence_all_none():
    assert avg_cadence([None, None]) is None


def test_cadence_below_threshold():
    cad = [55.0] * 50 + [85.0] * 50
    result = cadence_below_threshold_pct(cad, 65.0)
    assert result == pytest.approx(50.0, abs=1.0)


# ── HR stats ─────────────────────────────────────────────────────────────────

def test_avg_hr_basic():
    hr = [140.0, 145.0, 150.0, 155.0]
    assert avg_hr(hr) == pytest.approx(147.5, abs=0.1)


def test_max_hr_basic():
    hr = [140.0, 155.0, 180.0, 165.0]
    assert max_hr(hr) == 180.0


def test_hr_near_max_pct():
    # 50 points above 93% of 180 = 167.4 bpm, 50 below
    hr = [170.0] * 50 + [140.0] * 50
    result = hr_near_max_pct(hr, athlete_max_hr=180.0, threshold_pct=0.93)
    assert result == pytest.approx(50.0, abs=1.0)


# ── time_in_zones_from_stream ─────────────────────────────────────────────────

ZONE_DEFS = [
    {"zone": 1, "name": "Active Recovery", "power_pct_ftp": [0, 55]},
    {"zone": 2, "name": "Endurance", "power_pct_ftp": [56, 75]},
    {"zone": 3, "name": "Tempo", "power_pct_ftp": [76, 90]},
    {"zone": 4, "name": "Threshold / Sweet Spot", "power_pct_ftp": [91, 105]},
    {"zone": 5, "name": "VO2max", "power_pct_ftp": [106, 120]},
]


def test_zone_times_all_z2():
    ftp = 280.0
    watts = [180.0] * 100   # 64% FTP → Z2
    result = time_in_zones_from_stream(watts, ftp, ZONE_DEFS)
    assert result["Endurance"] == 100
    assert result["Active Recovery"] == 0


def test_zone_times_mixed():
    ftp = 280.0
    watts = [130.0] * 50 + [220.0] * 50  # Z1 then Z3
    result = time_in_zones_from_stream(watts, ftp, ZONE_DEFS)
    assert result["Active Recovery"] == 50
    assert result["Tempo"] == 50


def test_zone_times_skips_none():
    ftp = 280.0
    watts = [180.0, None, 180.0, None]
    result = time_in_zones_from_stream(watts, ftp, ZONE_DEFS)
    assert result["Endurance"] == 2
