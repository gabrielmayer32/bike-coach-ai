from __future__ import annotations
"""
Assembles a clean, already-computed session summary dict from raw API data.
This dict is what gets passed to the AI — never the raw streams.
"""

from datetime import date, timedelta
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict

from app.config import get_coaching_config
from app.intervals import client as icu
from app.metrics import compute


class SessionSummary(BaseModel):
    """Validated persisted interface between metric computation and analysis."""

    model_config = ConfigDict(extra="allow")

    summary_version: int
    activity_id: str
    athlete_id: str
    date: str
    name: str
    type: str
    is_indoor: bool
    ftp_W: Optional[float]
    ftp_source: Literal["activity", "sport_settings", "missing"]
    session_type: Optional[str] = None
    interval_source_policy: Literal["device_laps"] = "device_laps"
    interval_source: Literal[
        "device_laps", "device_laps_inferred", "missing"
    ] = "missing"
    interval_source_detail: Optional[
        Literal[
            "activity_laps",
            "icu_intervals_lap_mode",
            "icu_intervals_lap_count",
        ]
    ] = None
    device_lap_count: int = 0
    interval_source_verified: bool = False
    interval_metrics_available: bool = False
    automatic_interval_detection_used: Optional[bool] = None
    target_interval_membership_verified: bool = False
    rep_metrics_available: bool = False
    incomplete_data_reason: Optional[str] = None
    power_zone_times_s: dict[str, float]
    hr_zone_times_s: dict[str, float]
    time_in_target_zone_pct: Optional[float] = None
    interval_details: list[dict[str, Any]]


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

    is_indoor = bool(detail.get("trainer"))

    # ── Athlete baselines ──────────────────────────────────────────────────────
    # FTP is owned by Intervals.icu. The activity value preserves the setting
    # used when the ride was analysed; sport settings are the only fallback.
    activity_ftp = detail.get("icu_ftp")
    if activity_ftp:
        ftp = float(activity_ftp)
        ftp_source = "activity"
    else:
        intervals_profile = icu.get_athlete_profile(athlete_id)
        ftp = icu.configured_ftp_for_activity(
            intervals_profile,
            detail.get("type", ""),
            is_indoor,
        )
        ftp_source = "sport_settings" if ftp else "missing"
    lthr = (
        detail.get("lthr")
        or athlete_profile.get("lthr_bpm")
    )
    max_hr_val = (
        detail.get("athlete_max_hr")
        or athlete_profile.get("max_hr_bpm")
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
    if_val = None
    if ftp:
        if_val = (
            round(detail.get("icu_intensity", 0) / 100, 3)
            if detail.get("icu_intensity")
            else compute.intensity_factor(np_val, ftp)
        )
    tss = None
    if ftp:
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
    if zone_times_raw and ftp:
        zone_defs = cfg["zones"]["coggan_7"]
        for z_entry in zone_times_raw:
            zid = z_entry.get("id", "")
            if zid not in {f"Z{number}" for number in range(1, 8)}:
                continue
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

    hr_zone_times = (
        compute.time_in_hr_zones_from_stream(hr_s, lthr, cfg["zones"]["coggan_7"])
        if has_hr and lthr
        else {}
    )
    z2 = next((zone for zone in cfg["zones"]["coggan_7"] if zone["zone"] == 2), None)
    longest_hr_above_z2_s = None
    if has_hr and lthr and z2 and z2.get("hr_pct_lthr"):
        z2_hr_ceiling = lthr * z2["hr_pct_lthr"][1] / 100
        longest_hr_above_z2_s = compute.longest_consecutive_above(hr_s, z2_hr_ceiling)

    # ── Planned workout target extraction ─────────────────────────────────────
    target_power = _extract_target_power(planned)
    target_duration_s = _extract_target_duration(planned)
    step_targets = _extract_step_targets(planned)

    # ── Interval / rep breakdown ───────────────────────────────────────────────
    interval_list = _extract_interval_list(detail, streams, ftp, target_power)
    interval_source_detail = (
        _device_lap_source_detail(detail) if interval_list else None
    )
    interval_source = (
        "device_laps_inferred"
        if interval_source_detail == "icu_intervals_lap_count"
        else ("device_laps" if interval_source_detail else "missing")
    )
    work_intervals = [
        interval for interval in interval_list if interval.get("is_work") is True
    ]
    # A generic Intervals.icu WORK label does not prove that the boundary is a
    # prescribed work rep. Only establish target membership when resolved
    # planned steps map one-to-one to the returned work boundaries.
    if step_targets and len(step_targets) == len(work_intervals):
        for interval, interval_target in zip(work_intervals, step_targets):
            interval["is_target_work"] = True
            interval["target_membership_source"] = "planned_step_count"
            interval["planned_role"] = "work"
            interval["target_power_W"] = interval_target
            interval["power_compliance_pct"] = compute.power_compliance_pct(
                interval.get("avg_power_W"), interval_target
            )
    target_work_intervals = [
        interval
        for interval in interval_list
        if interval.get("is_target_work") is True
    ]
    target_membership_verified = bool(target_work_intervals)
    interval_powers = [
        interval.get("avg_power_W")
        for interval in target_work_intervals
        if interval.get("avg_power_W")
    ]

    fade = compute.rep_fade_pct(interval_powers) if len(interval_powers) >= 2 else None

    # ── Peak powers ────────────────────────────────────────────────────────────
    peak_5s = compute.peak_power_for_duration(power_s, 5) if has_power else None
    peak_10s = compute.peak_power_for_duration(power_s, 10) if has_power else None
    peak_15s = compute.peak_power_for_duration(power_s, 15) if has_power else None
    peak_30s = compute.peak_power_for_duration(power_s, 30) if has_power else None
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

    # ── Power compliance ──────────────────────────────────────────────────────
    interval_compliance = [
        abs(interval["power_compliance_pct"])
        for interval in target_work_intervals
        if interval.get("power_compliance_pct") is not None
    ]
    compliance = (
        round(sum(interval_compliance) / len(interval_compliance), 1)
        if interval_compliance else None
    )

    # ── RPE / feel ────────────────────────────────────────────────────────────
    rpe = detail.get("perceived_exertion") or detail.get("icu_rpe") or detail.get("feel")

    # ── Weather ───────────────────────────────────────────────────────────────
    temp_c = detail.get("average_weather_temp") or detail.get("average_temp")
    vo2_time_s = (
        compute.time_above_power_pct(power_s, ftp, 105)
        if has_power and ftp else None
    )

    # ── Assemble summary ──────────────────────────────────────────────────────
    summary: dict[str, Any] = {
        "summary_version": 5,
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
        "ftp_source": ftp_source,
        "if_value": if_val,
        "tss": tss,
        "vi": vi,
        # Power targets & compliance
        "target_power_W": target_power,
        "target_duration_s": target_duration_s,
        "power_compliance_pct": compliance,
        # Rep breakdown
        "interval_source": interval_source,
        "interval_source_policy": cfg["interval_analysis"]["source"],
        "interval_source_detail": interval_source_detail,
        "device_lap_count": _reported_device_lap_count(detail),
        "interval_source_verified": interval_source_detail in {
            "activity_laps", "icu_intervals_lap_mode"
        },
        "interval_metrics_available": bool(interval_list),
        "automatic_interval_detection_used": (
            None if interval_source_detail == "icu_intervals_lap_count" else False
        ),
        "target_interval_membership_verified": target_membership_verified,
        "rep_metrics_available": bool(target_work_intervals),
        "incomplete_data_reason": (
            "device_laps_not_available"
            if not interval_list
            else (
                None
                if target_membership_verified
                else "planned_interval_membership_not_verified"
            )
        ),
        "interval_count": len(work_intervals),
        "device_interval_count": len(interval_list),
        "target_interval_count": len(target_work_intervals),
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
        "power_zone_times_s": zone_times,
        "hr_zone_times_s": hr_zone_times,
        "time_in_target_zone_pct": None,
        "longest_hr_above_z2_s": longest_hr_above_z2_s,
        # Cadence
        "avg_cadence_rpm": avg_cad,
        # Peak powers
        "peak_5s_W": peak_5s,
        "peak_10s_W": peak_10s,
        "peak_15s_W": peak_15s,
        "peak_30s_W": peak_30s,
        "peak_60s_W": peak_60s,
        "peak_300s_W": peak_300s,
        "vo2_time_at_intensity_s": vo2_time_s,
        "vo2_time_at_intensity_pct": None,
        "sprint_vs_90d_best_pct": None,
        "torque_cadence_deviation_rpm": None,
        # RPE & subjective
        "rpe": rpe,
        # Environment
        "temp_c": temp_c,
        # Wellness context
        "wellness": _summarise_wellness(wellness),
        # Planned workout
        "planned_workout": _summarise_planned(planned),
    }

    SessionSummary.model_validate(summary)
    return summary


def _safe_mean(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 1) if clean else None


def _activity_uses_device_laps(detail: dict) -> bool:
    """Return true only when Intervals reports lap-based interval analysis."""
    for key in (
        "use_laps_for_power_intervals",
        "use_laps_for_intervals",
        "use_laps",
    ):
        if detail.get(key) is True:
            return True
    source = str(detail.get("interval_source") or "").strip().lower()
    return source in {"device_laps", "laps", "fit_laps", "activity_laps"}


def _reported_device_lap_count(detail: dict) -> int:
    """Return the source activity's lap count reported by Intervals.icu."""
    try:
        return max(0, int(detail.get("icu_lap_count") or 0))
    except (TypeError, ValueError):
        return 0


def _interval_duration_s(interval: dict) -> float | None:
    for key in ("elapsed_time", "moving_time", "duration_s"):
        value = interval.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    start = interval.get("start_index")
    end = interval.get("end_index")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return max(0.0, float(end) - float(start))
    return None


def _meaningful_intervals(intervals: Any) -> list[dict]:
    """Remove tiny recovery remnants without discarding short sprint laps."""
    if not isinstance(intervals, list):
        return []
    minimum = get_coaching_config()["interval_analysis"][
        "ignore_recovery_fragments_shorter_than_s"
    ]
    meaningful: list[dict] = []
    for interval in intervals:
        if not isinstance(interval, dict):
            continue
        duration = _interval_duration_s(interval)
        role_text = " ".join(
            str(interval.get(key) or "")
            for key in ("type", "lap_type", "intensity", "name", "label")
        ).lower()
        is_recovery = any(
            word in role_text
            for word in ("recovery", "rest", "warmup", "warm-up", "cooldown", "cool-down")
        )
        if duration is not None and duration < minimum and is_recovery:
            continue
        meaningful.append(interval)
    return meaningful


def _icu_intervals_correlate_with_device_laps(detail: dict) -> bool:
    """Recognise lap-derived intervals when the API omits its legacy lap-mode flag.

    Recent Intervals.icu activity payloads report ``icu_lap_count`` but do not
    always include ``use_laps_for_*``. In lap mode Intervals may omit recovery or
    otherwise non-meaningful laps, so the analysed interval count can be smaller
    than the source lap count. An explicit non-lap flag or auto-detected source
    always wins and keeps the intervals rejected.
    """
    for key in (
        "use_laps_for_power_intervals",
        "use_laps_for_intervals",
        "use_laps",
    ):
        if detail.get(key) is False:
            return False

    source = str(detail.get("interval_source") or "").strip().lower()
    if source in {"auto", "automatic", "detected", "icu_detected", "power"}:
        return False

    lap_count = _reported_device_lap_count(detail)
    intervals = _meaningful_intervals(detail.get("icu_intervals"))
    return (
        lap_count > 1
        and 0 < len(intervals) <= lap_count
    )


def _has_meaningful_device_laps(laps: Any) -> bool:
    """Ignore the single generic session lap present in many activity files."""
    laps = _meaningful_intervals(laps)
    if not laps:
        return False
    if len(laps) > 1:
        return True
    lap = laps[0]
    return any(
        lap.get(key) is not None
        for key in (
            "wkt_step_index",
            "workout_step_index",
            "intensity",
            "lap_type",
            "label",
            "name",
        )
    )


def _device_lap_source_detail(detail: dict) -> str | None:
    if _has_meaningful_device_laps(detail.get("laps")):
        return "activity_laps"
    intervals = detail.get("icu_intervals")
    if (
        _activity_uses_device_laps(detail)
        and isinstance(intervals, list)
        and intervals
        and isinstance(intervals[0], dict)
    ):
        return "icu_intervals_lap_mode"
    if _icu_intervals_correlate_with_device_laps(detail):
        return "icu_intervals_lap_count"
    return None


def _device_lap_payload(detail: dict) -> list[dict]:
    source = _device_lap_source_detail(detail)
    if source == "activity_laps":
        return _meaningful_intervals(detail["laps"])
    if source in {"icu_intervals_lap_mode", "icu_intervals_lap_count"}:
        return _meaningful_intervals(detail["icu_intervals"])
    return []


def _extract_interval_list(
    detail: dict,
    streams: dict,
    ftp: float | None,
    target_power: float | None,
) -> list[dict]:
    """Extract metrics only from verified device-lap boundaries."""

    watts_s = streams.get("watts", [])
    hr_s = streams.get("heartrate", [])
    cad_s = streams.get("cadence", [])
    torque_s = streams.get("torque", [])

    def _slice_stats(start: int, end: int) -> dict:
        w = [v for v in watts_s[start:end] if v is not None]
        h = [v for v in hr_s[start:end] if v is not None]
        c = [v for v in cad_s[start:end] if v is not None and v > 0]
        t = [v for v in torque_s[start:end] if v is not None]
        avg_w = round(sum(w) / len(w), 1) if w else None
        np_w = _normalised_power_slice(watts_s[start:end])
        vi = round(np_w / avg_w, 3) if np_w and avg_w else None
        return {
            "avg_power_W": avg_w,
            "np_W": np_w,
            "vi": vi,
            "avg_hr_bpm": round(sum(h) / len(h), 1) if h else None,
            "avg_cadence_rpm": round(sum(c) / len(c), 1) if c else None,
            "avg_torque_Nm": round(sum(t) / len(t), 1) if t else None,
            "within_rep_fade_pct": compute.within_rep_fade_pct(watts_s[start:end]),
            "peak_5s_W": compute.peak_power_for_duration(watts_s[start:end], 5),
            "peak_10s_W": compute.peak_power_for_duration(watts_s[start:end], 10),
            "peak_15s_W": compute.peak_power_for_duration(watts_s[start:end], 15),
            "peak_30s_W": compute.peak_power_for_duration(watts_s[start:end], 30),
        }

    def _is_work_lap(lap: dict) -> bool:
        if "is_work" in lap:
            return bool(lap["is_work"])
        intensity = lap.get("intensity")
        if isinstance(intensity, dict):
            intensity = (
                intensity.get("valueName")
                or intensity.get("value_name")
                or intensity.get("value")
            )
        text = " ".join(
            str(value or "") for value in (
                lap.get("name"), lap.get("label"), lap.get("type"),
                lap.get("lap_type"), intensity,
            )
        ).lower()
        return not any(word in text for word in ("warmup", "warm-up", "cooldown", "cool-down", "recovery", "rest"))

    # ── Device laps ────────────────────────────────────────────────────────────
    # Intervals calculations are accepted only when their boundaries have been
    # verified as device-lap based. Auto-detected intervals never reach here.
    laps = _device_lap_payload(detail)
    if laps and isinstance(laps, list) and isinstance(laps[0], dict):
        parsed = []
        cursor = 0
        for lap in laps:
            dur = lap.get("elapsed_time") or lap.get("moving_time") or 0
            start = int(lap.get("start_index", cursor) or cursor)
            end = int(lap.get("end_index", start + dur) or (start + dur))
            stats = _slice_stats(start, end) if watts_s else {}
            # Prefer API-provided metrics; fill gaps from stream slice
            avg_w = lap.get("average_watts") or lap.get("avg_power") or stats.get("avg_power_W")
            np_w = lap.get("normalized_power") or lap.get("weighted_average_watts") or stats.get("np_W")
            avg_hr = lap.get("average_heartrate") or lap.get("avg_hr") or stats.get("avg_hr_bpm")
            avg_cad = lap.get("average_cadence") or stats.get("avg_cadence_rpm")
            avg_torque = lap.get("average_torque") or stats.get("avg_torque_Nm")
            vi = round(np_w / avg_w, 3) if np_w and avg_w else None
            parsed.append({
                "start_index": start,
                "end_index": end,
                "duration_s": dur,
                "source": "device_lap",
                "source_detail": _device_lap_source_detail(detail),
                "label": lap.get("label") or lap.get("name"),
                "interval_type": lap.get("type") or lap.get("lap_type") or lap.get("intensity"),
                "is_work": _is_work_lap(lap),
                # WORK/RECOVERY here is Intervals' generic segmentation. It is
                # deliberately separate from membership in the prescribed set.
                "is_target_work": None,
                "target_membership_source": "unverified",
                "observed_role": "unknown",
                "observed_role_source": None,
                "planned_role": None,
                "avg_power_W": float(avg_w) if avg_w else None,
                "np_W": float(np_w) if np_w else None,
                "vi": vi,
                "avg_hr_bpm": avg_hr,
                "avg_cadence_rpm": avg_cad,
                "avg_torque_Nm": avg_torque,
                "decoupling_pct": lap.get("decoupling"),
                "training_load": lap.get("training_load"),
                "target_power_W": None,
                "power_compliance_pct": None,
                "within_rep_fade_pct": stats.get("within_rep_fade_pct"),
                "peak_5s_W": stats.get("peak_5s_W"),
                "peak_10s_W": stats.get("peak_10s_W"),
                "peak_15s_W": stats.get("peak_15s_W"),
                "peak_30s_W": stats.get("peak_30s_W"),
            })
            cursor = end
        return parsed

    # No device laps means no interval-level claims. In particular, do not parse
    # interval_summary or accept unverified icu_intervals as a quiet fallback.
    return []


def enrich_session_summary(summary: dict, session_type: str) -> dict:
    """Add metrics that depend on the selected configured session type."""
    cfg = get_coaching_config()
    session_cfg = cfg["session_types"][session_type]
    summary["session_type"] = session_type

    target_zone = session_cfg.get("target_zone")
    zone = next(
        (item for item in cfg["zones"]["coggan_7"] if item["zone"] == target_zone),
        None,
    )
    summary["time_in_target_zone_pct"] = (
        compute.time_in_target_zone_pct(
            summary.get("power_zone_times_s", {}), zone["name"]
        )
        if zone else None
    )

    if session_type == "recovery":
        sustained_high_hr = summary.get("longest_hr_above_z2_s")
        peak_30s = summary.get("peak_30s_W")
        ftp = summary.get("ftp_W")
        summary["recovery_aerobically_easy"] = (
            sustained_high_hr is not None
            and sustained_high_hr < 180
            and peak_30s is not None
            and ftp is not None
            and peak_30s < ftp
        )

    if session_type == "vo2max":
        work_duration = sum(
            interval.get("duration_s") or 0
            for interval in summary.get("interval_details", [])
            if interval.get("is_target_work") is True
        )
        intensity_time = summary.get("vo2_time_at_intensity_s")
        summary["vo2_time_at_intensity_pct"] = (
            round(min(intensity_time / work_duration * 100, 100), 1)
            if intensity_time is not None and work_duration > 0 else None
        )

    if session_type == "torque":
        cadence_values = [
            interval["avg_cadence_rpm"]
            for interval in summary.get("interval_details", [])
            if interval.get("is_target_work") is True
            and interval.get("avg_cadence_rpm")
        ]
        work_cadence = sum(cadence_values) / len(cadence_values) if cadence_values else None
        target_cadence = session_cfg.get("target_cadence_rpm")
        summary["torque_cadence_deviation_rpm"] = (
            round(abs(work_cadence - target_cadence), 1)
            if work_cadence is not None and target_cadence is not None else None
        )

    SessionSummary.model_validate(summary)
    return summary


def _normalised_power_slice(watts: list) -> float | None:
    """30-second rolling average NP for a stream slice."""
    clean = [w if w is not None else 0 for w in watts]
    if len(clean) < 30:
        return None
    window = 30
    rolling = [
        sum(clean[i:i + window]) / window
        for i in range(len(clean) - window + 1)
    ]
    np_val = (sum(r ** 4 for r in rolling) / len(rolling)) ** 0.25
    return round(np_val, 1)


def _extract_target_power(planned: dict | None) -> float | None:
    if not planned:
        return None
    # Intervals.icu planned workouts can embed target power in description or structured fields
    candidates = [planned, planned.get("workout_doc") or {}]
    for candidate in candidates:
        for key in ["target_watts", "watts"]:
            value = candidate.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
    return None


def _extract_step_targets(planned: dict | None) -> list[float]:
    """Extract resolved work-step watt targets without guessing phase positions."""
    if not planned:
        return []
    document = planned.get("workout_doc") or {}

    def target_from_step(step: dict) -> float | None:
        for key in ("target_watts", "watts", "power"):
            value = step.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
            if isinstance(value, (list, tuple)) and value:
                numeric = [float(item) for item in value if isinstance(item, (int, float))]
                if numeric:
                    return sum(numeric) / len(numeric)
            if isinstance(value, dict):
                numeric = [
                    float(value[subkey])
                    for subkey in ("value", "start", "end", "min", "max")
                    if isinstance(value.get(subkey), (int, float))
                ]
                if numeric:
                    return sum(numeric) / len(numeric)
        return None

    def walk(steps: list) -> list[float]:
        targets: list[float] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            nested = step.get("steps") or step.get("children")
            repeats = int(step.get("reps") or step.get("repeat") or 1)
            if isinstance(nested, list):
                targets.extend(walk(nested) * repeats)
                continue
            text = " ".join(str(step.get(key) or "") for key in ("name", "type")).lower()
            if any(word in text for word in ("warm", "cool", "recovery", "rest")):
                continue
            target = target_from_step(step)
            if target is not None:
                targets.extend([round(target, 1)] * repeats)
        return targets

    return walk(document.get("steps") or planned.get("steps") or [])


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
        "work_step_targets_W": _extract_step_targets(planned),
        "type": planned.get("type", ""),
    }
