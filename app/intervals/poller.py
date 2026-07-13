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

from app.config import get_settings
from app.db.session import get_session
from app.db import crud
from app.intervals import client as icu
from app.metrics.session_summary import build_session_summary
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
        summary = activity.session_summary_json
        similar = crud.get_similar_sessions(
            db, activity.athlete_id, activity.session_type, activity_id
        )

        ai_result = analyse_session(summary, athlete_dict, similar)
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
    Priority: activity name keywords (most reliable for this athlete population whose
    names encode the prescription) → physiological signals → IF fallback.
    """
    name = (summary.get("name") or "").lower()
    if_val = summary.get("if_value") or 0
    avg_cad = summary.get("avg_cadence_rpm") or 0

    # Ordered from most specific to least — first match wins.
    # Longer / more specific phrases must come before shorter ones that overlap.
    keyword_map = [
        # VO2 / SIT (supramaximal)
        ("sit",                  "vo2max"),
        ("vo2",                  "vo2max"),
        ("short intense",        "vo2max"),
        # Over-unders
        ("over under",           "over_unders"),
        ("over-under",           "over_unders"),
        # Torque / low-cadence
        ("torque",               "torque"),
        ("low cadence",          "torque"),
        ("force",                "torque"),
        # Threshold / sweet spot
        ("threshold",            "threshold"),
        ("sweet spot",           "threshold"),
        ("sweetspot",            "threshold"),
        ("lt2",                  "threshold"),
        ("ftp",                  "threshold"),
        # Tempo / LT1
        ("tempo",                "tempo"),
        ("lt1",                  "tempo"),
        ("z3",                   "tempo"),
        # Sprints / neuromuscular
        ("sprint",               "sprint_neuromuscular"),
        ("neuromuscular",        "sprint_neuromuscular"),
        ("torq+sprint",          "sprint_neuromuscular"),
        # Endurance / Z2
        ("aerobic endurance",    "endurance_z2"),
        ("endurance",            "endurance_z2"),
        ("zone 2",               "endurance_z2"),
        ("z2",                   "endurance_z2"),
        ("base",                 "endurance_z2"),
        # Recovery
        ("recovery",             "recovery"),
        ("easy",                 "recovery"),
        ("high cadence",         "recovery"),   # high-cadence recovery ride
        ("z1",                   "recovery"),
    ]
    for keyword, session_type in keyword_map:
        if keyword in name:
            return session_type

    # Low cadence sustained → torque session
    if avg_cad and 40 <= avg_cad <= 65 and if_val > 0.7:
        return "torque"

    # IF-based fallback when name gives no signal
    if if_val < 0.55:
        return "recovery"
    if if_val < 0.76:
        return "endurance_z2"
    if if_val < 0.88:
        return "tempo"
    if if_val < 1.05:
        return "threshold"
    if if_val < 1.20:
        return "vo2max"

    return "sprint_neuromuscular"
