from __future__ import annotations
"""
Scheduled poller — finds new activities and triggers analysis.

Structured so an Intervals.icu webhook can replace the polling loop later
without changing the core analysis logic: the webhook handler just calls
process_activity() directly.
"""

import logging
from datetime import date, timedelta

from apscheduler.schedulers.background import BackgroundScheduler

from app.config import get_coaching_config, get_settings
from app.db.session import get_session
from app.db import crud
from app.intervals import client as icu
from app.metrics.session_summary import build_session_summary, enrich_session_summary
from app.ai.analyser import analyse_session

log = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


# ── Core processing logic (also called directly by webhook) ───────────────────

def process_activity(athlete_id: str, activity_id: str, force: bool = False) -> dict | None:
    """
    Full pipeline for one activity:
    fetch → compute → AI analysis → store.
    Returns the analysis dict, or None if skipped.
    Pass force=True to bypass the already-processed guard (used for manual triggers).
    """
    db = get_session()
    try:
        athlete = crud.get_athlete(db, athlete_id)
        if not athlete:
            log.warning("Athlete %s not in DB — skipping activity %s", athlete_id, activity_id)
            return None

        if not force and crud.activity_exists(db, activity_id):
            log.debug("Activity %s already processed — skipping", activity_id)
            return None

        log.info("Processing activity %s for athlete %s", activity_id, athlete_id)

        # Build computed summary (all metrics in Python, no raw data to AI)
        athlete_dict = crud.athlete_as_dict(athlete)
        summary = build_session_summary(athlete_id, activity_id, athlete_dict)

        # Classify session type from activity name / type
        session_type = _classify_session(summary)
        enrich_session_summary(summary, session_type)
        if session_type == "sprint_neuromuscular":
            _add_sprint_benchmark(summary, athlete_id)

        # Athlete FTP is a display cache only and is always sourced from Intervals.
        if summary.get("ftp_source") != "missing" and summary.get("ftp_W"):
            athlete.ftp_W = summary["ftp_W"]

        # Fetch historical similar sessions for comparison context
        similar = crud.get_similar_sessions(db, athlete_id, session_type, activity_id)

        # Run AI analysis
        ai_result = analyse_session(summary, athlete_dict, similar)

        # Persist
        crud.store_activity(db, summary, session_type)
        crud.store_analysis(db, activity_id, athlete_id, ai_result)

        log.info(
            "Done: activity=%s verdict=%s escalate=%s cost=$%.4f",
            activity_id,
            ai_result.get("verdict"),
            ai_result.get("escalate_to_coach", {}).get("flag"),
            ai_result.get("_meta", {}).get("cost_usd", 0),
        )
        return ai_result

    except Exception:
        log.exception("Error processing activity %s", activity_id)
        return None
    finally:
        db.close()


def reanalyse_activity(activity_id: str) -> dict | None:
    """
    Re-run AI analysis on a stored activity using the current coaching_config.
    Does NOT re-fetch from Intervals.icu — uses the stored session_summary_json.
    Used for config tuning and comparison.
    """
    db = get_session()
    try:
        activity = db.query(crud.Activity).get(activity_id)
        if not activity or not activity.session_summary_json:
            log.warning("Activity %s not found or has no stored summary", activity_id)
            return None

        athlete = crud.get_athlete(db, activity.athlete_id)
        if not athlete:
            return None

        athlete_dict = crud.athlete_as_dict(athlete)
        # Rebuild facts from Intervals so old summaries cannot retain modeled,
        # local, or fabricated FTP values and missing metric fields.
        summary = build_session_summary(activity.athlete_id, activity_id, athlete_dict)
        session_type = _classify_session(summary)
        enrich_session_summary(summary, session_type)
        if session_type == "sprint_neuromuscular":
            _add_sprint_benchmark(summary, activity.athlete_id)
        if summary.get("ftp_source") != "missing" and summary.get("ftp_W"):
            athlete.ftp_W = summary["ftp_W"]
        similar = crud.get_similar_sessions(
            db, activity.athlete_id, session_type, activity_id
        )

        ai_result = analyse_session(summary, athlete_dict, similar)
        crud.store_activity(db, summary, session_type)
        crud.store_analysis(db, activity_id, activity.athlete_id, ai_result)

        log.info("Reanalysis complete for activity %s verdict=%s", activity_id, ai_result.get("verdict"))
        return ai_result

    except Exception:
        log.exception("Error reanalysing activity %s", activity_id)
        return None
    finally:
        db.close()


# ── Polling job ────────────────────────────────────────────────────────────────

def _poll_all_athletes() -> None:
    """Check all active athletes for new activities in the last 2 days."""
    db = get_session()
    try:
        athletes = crud.list_active_athletes(db)
        since = date.today() - timedelta(days=2)
        log.debug("Polling %d athletes since %s", len(athletes), since)

        for athlete in athletes:
            try:
                activities = icu.list_activities(athlete.id, since=since)
                for act in activities:
                    act_id = act.get("id")
                    if act_id and not crud.activity_exists(db, str(act_id)):
                        process_activity(athlete.id, str(act_id))
            except Exception:
                log.exception("Error polling athlete %s", athlete.id)
    finally:
        db.close()


# ── Scheduler lifecycle ────────────────────────────────────────────────────────

def start_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        return

    interval = get_settings().poll_interval_seconds
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        _poll_all_athletes,
        trigger="interval",
        seconds=interval,
        id="poll_athletes",
        replace_existing=True,
        max_instances=1,
    )
    _scheduler.start()
    log.info("Poller started — interval=%ds", interval)


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        log.info("Poller stopped")


# ── Session type classifier ────────────────────────────────────────────────────

def _classify_session(summary: dict) -> str:
    """
    Heuristic classifier — maps an activity to one of the coaching_config session_types.
    Priority: planned workout text → activity name → physiological signals → IF fallback.
    """
    cfg = get_coaching_config()["classification"]
    planned = summary.get("planned_workout") or {}
    text = " ".join(
        str(value or "")
        for value in (
            planned.get("name"),
            planned.get("description"),
            summary.get("name"),
        )
    ).lower()
    if_val = summary.get("if_value") or 0
    avg_cad = summary.get("avg_cadence_rpm") or 0

    for rule in cfg["keyword_priority"]:
        if any(keyword.lower() in text for keyword in rule["keywords"]):
            return rule["session_type"]

    # Low cadence sustained → torque session
    if (
        avg_cad
        and cfg["low_cadence_min_rpm"] <= avg_cad <= cfg["low_cadence_max_rpm"]
        and if_val > cfg["low_cadence_min_if"]
    ):
        return cfg["low_cadence_session_type"]

    # IF-based fallback when name gives no signal
    for rule in cfg["if_fallback"]:
        if rule.get("below") is not None and if_val < rule["below"]:
            return rule["session_type"]

    return cfg["default_session_type"]


def _add_sprint_benchmark(summary: dict, athlete_id: str) -> None:
    """Compare current sprint peaks with Intervals' rolling 90-day curve."""
    curve = icu.get_power_curve_range(
        athlete_id,
        since=date.today() - timedelta(days=90),
    )
    seconds = curve.get("secs", [])
    watts = curve.get("watts", [])
    comparisons: dict[str, dict] = {}
    ratios: list[float] = []
    for duration in (5, 10, 15):
        try:
            index = seconds.index(duration)
            best = watts[index]
        except (ValueError, IndexError):
            best = None
        current = summary.get(f"peak_{duration}s_W")
        ratio = round(current / best * 100, 1) if current and best else None
        comparisons[f"{duration}s"] = {
            "current_W": current,
            "best_90d_W": best,
            "pct_of_best": ratio,
        }
        if ratio is not None:
            ratios.append(ratio)
    summary["sprint_90d_comparison"] = comparisons
    summary["sprint_vs_90d_best_pct"] = min(ratios) if ratios else None
