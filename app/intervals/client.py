from __future__ import annotations
"""
Intervals.icu API client.

Key findings from the spike:
- Auth: HTTP Basic, username="API_KEY", password=<key>
- Athlete roster: GET /athlete/0/athlete-summary.json  → fields are athlete_id / athlete_name
- Activity list:  GET /athlete/{id}/activities
- Activity detail: GET /athlete/{id}/activities/{act_id}  → returns list of 1
- Streams:  GET /activity/{act_id}/streams  (no athlete prefix — different path)
- Calendar: GET /athlete/{id}/events
- Wellness:  GET /athlete/{id}/wellness/{date}
- Strava activities: source=STRAVA, streams return 422 — skip them
"""

import logging
import time
from datetime import date, timedelta
from typing import Any

import httpx

from app.config import get_settings

log = logging.getLogger(__name__)

BASE_URL = "https://intervals.icu/api/v1"
_MIN_REQUEST_GAP = 0.12          # ~8 req/s, well under their 10 req/s limit
_last_call: float = 0.0


def _auth() -> tuple[str, str]:
    return ("API_KEY", get_settings().intervals_api_key)


def _get(path: str, params: dict | None = None) -> Any:
    """Rate-limited authenticated GET. Returns parsed JSON."""
    global _last_call
    gap = time.monotonic() - _last_call
    if gap < _MIN_REQUEST_GAP:
        time.sleep(_MIN_REQUEST_GAP - gap)

    url = f"{BASE_URL}{path}"
    log.debug("GET %s params=%s", url, params)
    resp = httpx.get(url, auth=_auth(), params=params, timeout=20)
    _last_call = time.monotonic()

    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


# ── Athlete roster ─────────────────────────────────────────────────────────────

def list_athletes() -> list[dict]:
    """Return all coached athletes (from the coach account's summary endpoint)."""
    data = _get("/athlete/0/athlete-summary.json") or []
    return [
        {
            "id": a["athlete_id"],
            "name": a.get("athlete_name", ""),
            "eftp": a.get("eftp"),
            "fitness": a.get("fitness"),
            "fatigue": a.get("fatigue"),
            "form": a.get("form"),
            "weight": a.get("weight"),
        }
        for a in data
        if a.get("athlete_id")
    ]


# ── Activities ─────────────────────────────────────────────────────────────────

def list_activities(
    athlete_id: str,
    since: date,
    until: date | None = None,
) -> list[dict]:
    """Return activity stubs for an athlete in a date range. Skips Strava activities."""
    until = until or date.today()
    data = _get(
        f"/athlete/{athlete_id}/activities",
        params={"oldest": since.isoformat(), "newest": until.isoformat()},
    ) or []
    return [a for a in data if a.get("source") != "STRAVA"]


def get_activity_detail(athlete_id: str, activity_id: str) -> dict | None:
    """Return full activity detail. The API returns a list of 1."""
    data = _get(f"/athlete/{athlete_id}/activities/{activity_id}")
    if not data:
        return None
    return data[0] if isinstance(data, list) else data


def get_activity_streams(activity_id: str) -> list[dict]:
    """
    Return raw streams for an activity.
    Correct endpoint: /activity/{id}/streams  (no athlete prefix).
    Returns list of {type, data} dicts.
    """
    data = _get(f"/activity/{activity_id}/streams")
    return data or []


def streams_as_dict(streams: list[dict]) -> dict[str, list]:
    """Convert the streams list into {stream_type: [values]} for easy access."""
    return {s["type"]: s.get("data", []) for s in streams}


# ── Calendar / planned workouts ────────────────────────────────────────────────

def get_events_for_date(athlete_id: str, on_date: date) -> list[dict]:
    """Return calendar events (planned workouts) for a single date."""
    data = _get(
        f"/athlete/{athlete_id}/events",
        params={"oldest": on_date.isoformat(), "newest": on_date.isoformat()},
    )
    return data or []


def get_planned_workout(athlete_id: str, activity_detail: dict) -> dict | None:
    """
    Try to find the planned workout that matches a completed activity.
    Uses paired_event_id if present, otherwise scans events for that date.
    """
    paired_id = activity_detail.get("paired_event_id")
    activity_date = str(activity_detail.get("start_date_local", ""))[:10]
    if not activity_date:
        return None

    try:
        on_date = date.fromisoformat(activity_date)
    except ValueError:
        return None

    events = get_events_for_date(athlete_id, on_date)
    if not events:
        return None

    if paired_id:
        for e in events:
            if str(e.get("id")) == str(paired_id):
                return e

    # Fall back: first workout-type event on that date
    for e in events:
        if e.get("type") == "Workout" or e.get("category") == "WORKOUT":
            return e

    return None


# ── Wellness ───────────────────────────────────────────────────────────────────

def get_wellness(athlete_id: str, on_date: date) -> dict | None:
    """
    Return wellness data for a date. Returns None on 404.
    Key fields: restingHR, sleepSecs, sleepScore, ctl, atl, rampRate, weight,
                hrvScore (None if athlete doesn't track HRV).
    """
    return _get(f"/athlete/{athlete_id}/wellness/{on_date.isoformat()}")


def get_wellness_range(athlete_id: str, since: date, until: date | None = None) -> list[dict]:
    """Return daily CTL/ATL/TSB/rampRate entries for a date range."""
    until = until or date.today()
    rows = _get(
        f"/athlete/{athlete_id}/wellness",
        params={"oldest": since.isoformat(), "newest": until.isoformat()},
    ) or []
    result = []
    for r in rows:
        ctl = r.get("ctl")
        atl = r.get("atl")
        result.append({
            "date": r["id"],
            "ctl": round(ctl, 1) if ctl else None,
            "atl": round(atl, 1) if atl else None,
            "tsb": round(ctl - atl, 1) if ctl and atl else None,
            "ramp_rate": round(r["rampRate"], 1) if r.get("rampRate") else None,
        })
    return result


def get_power_curve_range(athlete_id: str, since: date, until: date | None = None) -> dict:
    """
    Return the best-effort power curve for a date range.
    Returns {secs: [...], watts: [...], w_per_kg: [...], ftp: int|None}.
    """
    until = until or date.today()
    data = _get(
        f"/athlete/{athlete_id}/power-curves",
        params={
            "oldest": since.isoformat(),
            "newest": until.isoformat(),
            "type": "Ride",
        },
    )
    if not data or not data.get("list"):
        return {"secs": [], "watts": [], "w_per_kg": [], "ftp": None}

    curve = data["list"][0]
    ftp = None
    for model in (curve.get("powerModels") or []):
        if model.get("ftp"):
            ftp = model["ftp"]
            break

    return {
        "secs": curve.get("secs", []),
        "watts": curve.get("watts", []),
        "w_per_kg": curve.get("watts_per_kg", []),
        "ftp": ftp,
    }


# ── Athlete profile / power curve ─────────────────────────────────────────────

def get_athlete_profile(athlete_id: str) -> dict | None:
    """Return the athlete's own profile (FTP, max HR, weight, zones)."""
    data = _get(f"/athlete/{athlete_id}/profile")
    if not data:
        return None
    return data.get("athlete", data)


def get_power_curve(athlete_id: str, start: date | None = None) -> dict | None:
    """
    Return the athlete's power curve (best efforts by duration).
    Used to compare sprint/VO2 peaks against rolling 90-day bests.
    """
    params: dict = {}
    if start:
        params["oldest"] = start.isoformat()
    data = _get(f"/athlete/{athlete_id}/power_curves", params=params or None)
    return data
