from __future__ import annotations
"""
Assembles a clean, already-computed session summary dict from raw API data.
This dict is what gets passed to the AI — never the raw streams.
"""

from datetime import date, timedelta
from typing import Any

from app.config import get_coaching_config
from app.intervals import client as icu
from app.metrics import compute


def build_session_summary(
    athlete_id: str,
    activity_id: str,
    athlete_profile: dict,
) -> dict:
    """
    Fetch all data for one activity and return a structured, computed summary.
    The AI sees only this dict — no raw streams.
    """
    cfg = get_coaching_config()

    # ── Raw fetch ──────────────────────────────────────────────────────────────
    detail = icu.get_activity_detail(athlete_id, activity_id)
    if not detail:
        raise ValueError(f"Activity {activity_id} not found for athlete {athlete_id}")

    streams_raw = icu.get_activity_streams(activity_id)
    streams = icu.streams_as_dict(streams_raw)

    activity_date_str = str(detail.get("start_date_local", ""))[:10]
    try:
        activity_date = date.fromisoformat(activity_date_str)
    except ValueError:
        activity_date = date.today()

    planned = icu.get_planned_workout(athlete_id, detail)
    wellness = icu.get_wellness(athlete_id, activity_date)

    # ── Athlete baselines ──────────────────────────────────────────────────────
    ftp = (
        detail.get("icu_ftp")
        or detail.get("icu_pm_ftp_watts")
        or athlete_profile.get("ftp")
        or 250  # fallback so maths don't break
    )
    lthr = (
        detail.get("lthr")
        or athlete_profile.get("lthr")
    )
    max_hr_val = (
        detail.get("athlete_max_hr")
        or athlete_profile.get("max_hr")
    )

    # ── Stream extraction ──────────────────────────────────────────────────────
    power_s: list = streams.get("watts", [])
    hr_s: list = streams.get("heartrate", [])
    cadence_s: list = streams.get("cadence", [])
    time_s: list = streams.get("time", [])
    torque_s: list = streams.get("torque", [])

    has_power = any(v is not None for v in power_s)
    has_hr = any(v is not None for v in hr_s)

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    # Prefer Intervals.icu pre-computed values; fall back to our own calculation.
    avg_power = detail.get("icu_average_watts") or (
        _safe_mean(power_s) if has_power else None
    )
    np_val = detail.get("icu_weighted_avg_watts") or (
        compute.normalised_power(power_s) if has_power else None
    )
    if_val = (
        round(detail.get("icu_intensity", 0) / 100, 3)
        if detail.get("icu_intensity")
        else compute.intensity_factor(np_val, ftp)
    )
    tss = detail.get("icu_training_load") or compute.training_stress_score(
        detail.get("moving_time", 0), if_val
    )
    vi = detail.get("icu_variability_index") or compute.variability_index(np_val, avg_power)
    decoupling = detail.get("decoupling") or (
        compute.pw_hr_decoupling(power_s, hr_s) if has_power and has_hr else None
    )
    kj = round(detail.get("icu_joules", 0) / 1000, 1) if detail.get("icu_joules") else None

    # ── Zone times ────────────────────────────────────────────────────────────
    zone_times_raw = detail.get("icu_zone_times", [])
    zone_times: dict[str, float] = {}
    if zone_times_raw:
        zone_defs = cfg["zones"]["coggan_7"]
        for z_entry in zone_times_raw:
            zid = z_entry.get("id", "")
            secs = z_entry.get("secs", 0)
            # Map Z1..Z7 to zone names
            matched = next(
                (z["name"] for z in zone_defs if f"Z{z['zone']}" == zid), zid
            )
            zone_times[matched] = secs
    elif has_power and ftp:
        zone_times = compute.time_in_zones_from_stream(
            power_s, ftp, cfg["zones"]["coggan_7"]
        )

    # ── Interval / rep breakdown ───────────────────────────────────────────────
    interval_list = _extract_interval_list(detail, streams, ftp)
    interval_powers = [iv.get("avg_power_W") for iv in interval_list if iv.get("avg_power_W")]

    fade = compute.rep_fade_pct(interval_powers) if len(interval_powers) >= 2 else None

    # ── Peak powers ────────────────────────────────────────────────────────────
    peak_5s = compute.peak_power_for_duration(power_s, 5) if has_power else None
    peak_60s = compute.peak_power_for_duration(power_s, 60) if has_power else None
    peak_300s = compute.peak_power_for_duration(power_s, 300) if has_power else None

    # ── HR stats ──────────────────────────────────────────────────────────────
    avg_hr_val = compute.avg_hr(hr_s) if has_hr else detail.get("average_heartrate")
    max_hr_measured = compute.max_hr(hr_s) if has_hr else detail.get("max_heartrate")
    hr_near_max = (
        compute.hr_near_max_pct(hr_s, max_hr_val)
        if has_hr and max_hr_val
        else None
    )

    # ── Cadence ───────────────────────────────────────────────────────────────
    avg_cad = compute.avg_cadence(cadence_s) if cadence_s else detail.get("average_cadence")

    # ── Planned workout target extraction ─────────────────────────────────────
    target_power = _extract_target_power(planned)
    target_duration_s = _extract_target_duration(planned)

    # ── Power compliance ──────────────────────────────────────────────────────
    compliance = (
        compute.power_compliance_pct(avg_power, target_power)
        if avg_power and target_power
        else None
    )

    # ── RPE / feel ────────────────────────────────────────────────────────────
    rpe = detail.get("perceived_exertion") or detail.get("icu_rpe") or detail.get("feel")

    # ── Weather ───────────────────────────────────────────────────────────────
    temp_c = detail.get("average_weather_temp") or detail.get("average_temp")
    is_indoor = bool(detail.get("trainer"))

    # ── Assemble summary ──────────────────────────────────────────────────────
    summary: dict[str, Any] = {
        # Identity
        "activity_id": activity_id,
        "athlete_id": athlete_id,
        "date": activity_date_str,
        "name": detail.get("name", ""),
        "type": detail.get("type", ""),
        "source": detail.get("source", ""),
        "is_indoor": is_indoor,
        # Duration & load
        "duration_s": detail.get("moving_time", 0),
        "elapsed_s": detail.get("elapsed_time", 0),
        "distance_m": detail.get("distance"),
        "elevation_m": detail.get("total_elevation_gain"),
        "kj": kj,
        # Power aggregate
        "avg_power_W": avg_power,
        "np_W": np_val,
        "ftp_W": ftp,
        "if_value": if_val,
        "tss": tss,
        "vi": vi,
        # Power targets & compliance
        "target_power_W": target_power,
        "target_duration_s": target_duration_s,
        "power_compliance_pct": compliance,
        # Rep breakdown
        "interval_count": len(interval_list),
        "interval_powers_W": interval_powers,
        "interval_details": interval_list,
        "rep_fade_pct": fade,
        # HR
        "avg_hr_bpm": avg_hr_val,
        "max_hr_bpm": max_hr_measured,
        "lthr_bpm": lthr,
        "athlete_max_hr": max_hr_val,
        "hr_near_max_pct": hr_near_max,
        # Decoupling & zones
        "decoupling_pct": decoupling,
        "zone_times_s": zone_times,
        # Cadence
        "avg_cadence_rpm": avg_cad,
        # Peak powers
        "peak_5s_W": peak_5s,
        "peak_60s_W": peak_60s,
        "peak_300s_W": peak_300s,
        # RPE & subjective
        "rpe": rpe,
        # Environment
        "temp_c": temp_c,
        # Wellness context
        "wellness": _summarise_wellness(wellness),
        # Planned workout
        "planned_workout": _summarise_planned(planned),
    }

    return summary


def _safe_mean(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 1) if clean else None


def _extract_interval_list(
    detail: dict, streams: dict, ftp: float
) -> list[dict]:
    """
    Attempt to extract per-interval metrics from lap markers.
    interval_summary from Intervals.icu is a list of human-readable strings
    (e.g. "1x 4m40s 210w") — not structured. We parse those into lightweight
    dicts for the AI's context rather than stream-sliced details.
    Falls back to empty list if nothing useful is available.
    """
    interval_summary = detail.get("interval_summary")
    if not interval_summary:
        return []

    # If it's a list of strings like ["1x 86s 214w", ...], parse them
    items = interval_summary if isinstance(interval_summary, list) else []
    if not items or not isinstance(items[0], str):
        return []

    parsed = []
    import re
    for entry in items:
        # Pattern: "Nx Ym Zs Pw" or "Nx Zs Pw" — flexible
        m = re.search(r'(\d+)x\s+((?:\d+m)?\s*(?:\d+s)?)\s+(\d+)w', entry)
        if not m:
            continue
        count = int(m.group(1))
        duration_str = m.group(2).strip()
        power_w = int(m.group(3))

        # Parse duration
        dur_s = 0
        mm = re.search(r'(\d+)m', duration_str)
        ss = re.search(r'(\d+)s', duration_str)
        if mm:
            dur_s += int(mm.group(1)) * 60
        if ss:
            dur_s += int(ss.group(1))

        for _ in range(count):
            parsed.append({
                "duration_s": dur_s,
                "avg_power_W": float(power_w),
                "np_W": None,
                "avg_hr_bpm": None,
                "avg_cadence_rpm": None,
                "vi": None,
            })

    return parsed


def _extract_target_power(planned: dict | None) -> float | None:
    if not planned:
        return None
    # Intervals.icu planned workouts can embed target power in description or structured fields
    for key in ["target_watts", "load_target", "watts"]:
        if planned.get(key):
            return float(planned[key])
    return None


def _extract_target_duration(planned: dict | None) -> int | None:
    if not planned:
        return None
    for key in ["moving_time", "duration", "time"]:
        if planned.get(key):
            return int(planned[key])
    return None


def _summarise_wellness(wellness: dict | None) -> dict:
    if not wellness:
        return {}
    return {
        "resting_hr_bpm": wellness.get("restingHR"),
        "hrv_score": wellness.get("hrvScore"),
        "sleep_secs": wellness.get("sleepSecs"),
        "sleep_score": wellness.get("sleepScore"),
        "ctl": wellness.get("ctl"),
        "atl": wellness.get("atl"),
        "tsb": (
            round(wellness["ctl"] - wellness["atl"], 1)
            if wellness.get("ctl") and wellness.get("atl")
            else None
        ),
        "ramp_rate": wellness.get("rampRate"),
        "weight_kg": wellness.get("weight"),
        "mood": wellness.get("mood"),
        "fatigue": wellness.get("fatigue"),
        "motivation": wellness.get("motivation"),
        "soreness": wellness.get("soreness"),
    }


def _summarise_planned(planned: dict | None) -> dict:
    if not planned:
        return {}
    return {
        "name": planned.get("name", ""),
        "description": (planned.get("description") or "")[:500],
        "target_power_W": _extract_target_power(planned),
        "target_duration_s": _extract_target_duration(planned),
        "type": planned.get("type", ""),
    }
