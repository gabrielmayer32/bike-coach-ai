from __future__ import annotations
"""
Deterministic metric computation.

The app computes the facts. The AI does the judgement.
Never pass raw streams to the LLM — everything here is computed in Python.
"""

import math
from typing import Any


# ── Normalised Power ───────────────────────────────────────────────────────────

def normalised_power(watts: list[float | None], smoothing_window: int = 30) -> float | None:
    """
    Standard NP algorithm: 30s rolling average → raise to 4th power → mean → 4th root.
    Returns None if the stream is empty or all-null.
    """
    clean = [w for w in watts if w is not None]
    if len(clean) < smoothing_window:
        return None

    # 30-second rolling average (1Hz assumed)
    rolled: list[float] = []
    for i in range(smoothing_window - 1, len(clean)):
        window = clean[i - smoothing_window + 1 : i + 1]
        rolled.append(sum(window) / smoothing_window)

    if not rolled:
        return None

    mean_4th = sum(v**4 for v in rolled) / len(rolled)
    return round(mean_4th**0.25, 1)


# ── Intensity Factor & TSS ─────────────────────────────────────────────────────

def intensity_factor(np_watts: float, ftp: float) -> float | None:
    if not ftp or not np_watts:
        return None
    return round(np_watts / ftp, 3)


def training_stress_score(duration_secs: float, if_value: float) -> float | None:
    if not duration_secs or not if_value:
        return None
    return round((duration_secs * if_value**2) / 3600 * 100, 1)


# ── Variability Index ──────────────────────────────────────────────────────────

def variability_index(np_watts: float | None, avg_watts: float | None) -> float | None:
    if not np_watts or not avg_watts or avg_watts == 0:
        return None
    return round(np_watts / avg_watts, 3)


# ── HR / Power Decoupling (Pw:Hr) ─────────────────────────────────────────────

def pw_hr_decoupling(
    power_stream: list[float | None],
    hr_stream: list[float | None],
) -> float | None:
    """
    Aerobic decoupling: compare efficiency factor (avg_power / avg_hr) in the
    first half of the ride vs the second half. Returns % change.
    A positive value means EF degraded (HR drifted up relative to power).
    """
    if not power_stream or not hr_stream:
        return None

    pairs = [
        (p, h)
        for p, h in zip(power_stream, hr_stream)
        if p is not None and h is not None and h > 0
    ]
    if len(pairs) < 60:
        return None

    mid = len(pairs) // 2
    first_half = pairs[:mid]
    second_half = pairs[mid:]

    ef1 = _efficiency_factor(first_half)
    ef2 = _efficiency_factor(second_half)

    if ef1 is None or ef2 is None or ef1 == 0:
        return None

    # Positive = EF degraded (HR drifted up relative to power — bad aerobic sign)
    # Negative = EF improved (HR came down — unlikely but possible with warm-up)
    return round((ef1 - ef2) / ef1 * 100, 1)


def _efficiency_factor(pairs: list[tuple[float, float]]) -> float | None:
    if not pairs:
        return None
    avg_power = sum(p for p, _ in pairs) / len(pairs)
    avg_hr = sum(h for _, h in pairs) / len(pairs)
    if avg_hr == 0:
        return None
    return avg_power / avg_hr


# ── Time in Zone ──────────────────────────────────────────────────────────────

def time_in_zones_from_stream(
    power_stream: list[float | None],
    ftp: float,
    zone_boundaries: list[dict],
) -> dict[str, float]:
    """
    Compute seconds spent in each Coggan zone from raw power stream.
    zone_boundaries from coaching_config zones.coggan_7 list.
    Returns {zone_name: seconds}.
    """
    result = {z["name"]: 0.0 for z in zone_boundaries}
    for w in power_stream:
        if w is None:
            continue
        pct = (w / ftp) * 100
        for zone in zone_boundaries:
            lo, hi = zone["power_pct_ftp"]
            if lo <= pct <= hi:
                result[zone["name"]] += 1.0
                break
    return result


def time_in_target_zone_pct(
    zone_times_secs: dict[str, float],
    target_zone_name: str,
) -> float | None:
    total = sum(zone_times_secs.values())
    if total == 0:
        return None
    target = zone_times_secs.get(target_zone_name, 0.0)
    return round(target / total * 100, 1)


# ── Interval Analysis ─────────────────────────────────────────────────────────

def extract_intervals_from_streams(
    time_stream: list[int],
    power_stream: list[float | None],
    hr_stream: list[float | None],
    cadence_stream: list[float | None],
    lap_markers: list[dict] | None = None,
) -> list[dict]:
    """
    If lap_markers are available (from the activity detail's lap data), use them
    to slice the streams into interval segments. Otherwise returns empty list.
    Each returned dict has: start_s, end_s, avg_power, np, avg_hr, avg_cadence, duration_s.
    """
    if not lap_markers:
        return []

    intervals = []
    for lap in lap_markers:
        start_s = lap.get("start_index", 0)
        end_s = lap.get("end_index", len(time_stream) - 1)

        p_slice = [power_stream[i] for i in range(start_s, end_s + 1) if i < len(power_stream)]
        h_slice = [hr_stream[i] for i in range(start_s, end_s + 1) if i < len(hr_stream)] if hr_stream else []
        c_slice = [cadence_stream[i] for i in range(start_s, end_s + 1) if i < len(cadence_stream)] if cadence_stream else []

        clean_p = [v for v in p_slice if v is not None]
        clean_h = [v for v in h_slice if v is not None]
        clean_c = [v for v in c_slice if v is not None]

        avg_p = round(sum(clean_p) / len(clean_p), 1) if clean_p else None
        np_val = normalised_power(p_slice)
        avg_h = round(sum(clean_h) / len(clean_h), 1) if clean_h else None
        avg_c = round(sum(clean_c) / len(clean_c), 1) if clean_c else None

        intervals.append({
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": end_s - start_s,
            "avg_power_W": avg_p,
            "np_W": np_val,
            "avg_hr_bpm": avg_h,
            "avg_cadence_rpm": avg_c,
            "vi": variability_index(np_val, avg_p),
        })
    return intervals


def rep_fade_pct(interval_powers: list[float | None]) -> float | None:
    """
    % drop from first rep's power to last rep's power.
    Positive = faded (got weaker). Negative = negative split (built).
    """
    valid = [p for p in interval_powers if p is not None]
    if len(valid) < 2:
        return None
    first, last = valid[0], valid[-1]
    if first == 0:
        return None
    return round((first - last) / first * 100, 1)


def within_rep_fade_pct(power_slice: list[float | None]) -> float | None:
    """
    Compare first half vs second half avg power within a single rep.
    Positive = second half was weaker (went out too hard).
    """
    clean = [p for p in power_slice if p is not None]
    if len(clean) < 20:
        return None
    mid = len(clean) // 2
    first_half_avg = sum(clean[:mid]) / mid
    second_half_avg = sum(clean[mid:]) / (len(clean) - mid)
    if first_half_avg == 0:
        return None
    return round((first_half_avg - second_half_avg) / first_half_avg * 100, 1)


# ── Power Compliance ──────────────────────────────────────────────────────────

def power_compliance_pct(actual_avg_W: float | None, target_W: float | None) -> float | None:
    """
    % deviation of actual average power from target.
    Positive = over target, negative = under target.
    Absolute value used for well/okay/poor verdict.
    """
    if actual_avg_W is None or not target_W:
        return None
    return round((actual_avg_W - target_W) / target_W * 100, 1)


# ── Peak Power from streams ────────────────────────────────────────────────────

def peak_power_for_duration(watts: list[float | None], duration_s: int) -> float | None:
    """Best average power for a given duration (seconds) from the stream."""
    clean = [w if w is not None else 0.0 for w in watts]
    if len(clean) < duration_s:
        return None
    best = 0.0
    for i in range(len(clean) - duration_s + 1):
        window_avg = sum(clean[i : i + duration_s]) / duration_s
        if window_avg > best:
            best = window_avg
    return round(best, 1)


# ── Cadence stats ──────────────────────────────────────────────────────────────

def avg_cadence(cadence_stream: list[float | None]) -> float | None:
    clean = [c for c in cadence_stream if c is not None and c > 0]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 1)


def cadence_below_threshold_pct(
    cadence_stream: list[float | None], threshold_rpm: float
) -> float | None:
    """% of pedalling time where cadence was below threshold (for torque sessions)."""
    pedalling = [c for c in cadence_stream if c is not None and c > 0]
    if not pedalling:
        return None
    below = sum(1 for c in pedalling if c < threshold_rpm)
    return round(below / len(pedalling) * 100, 1)


# ── Heart rate stats ───────────────────────────────────────────────────────────

def avg_hr(hr_stream: list[float | None]) -> float | None:
    clean = [h for h in hr_stream if h is not None and h > 0]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 1)


def max_hr(hr_stream: list[float | None]) -> float | None:
    clean = [h for h in hr_stream if h is not None]
    return max(clean) if clean else None


def hr_near_max_pct(hr_stream: list[float | None], athlete_max_hr: float, threshold_pct: float = 0.93) -> float | None:
    """% of time HR was above threshold_pct of max HR (useful for VO2max sessions)."""
    clean = [h for h in hr_stream if h is not None and h > 0]
    if not clean or not athlete_max_hr:
        return None
    threshold = athlete_max_hr * threshold_pct
    above = sum(1 for h in clean if h >= threshold)
    return round(above / len(clean) * 100, 1)
