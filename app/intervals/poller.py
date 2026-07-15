from __future__ import annotations
"""
Scheduled poller — finds new activities and triggers analysis.

Structured so an Intervals.icu webhook can replace the polling loop later
without changing the core analysis logic: the webhook handler just calls
process_activity() directly.
"""

import logging
import re
from datetime import date, timedelta
from itertools import combinations

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
    Map an activity using coach-owned recognition signatures.

    Priority: planned workout text, completed title, observed interval signatures,
    configured whole-activity IF ranges, then the honest unclassified fallback.
    """
    cfg = get_coaching_config()
    classification = cfg["classification"]
    session_types = sorted(
        cfg["session_types"].items(),
        key=lambda item: item[1]["recognition"]["priority"],
    )
    planned = summary.get("planned_workout") or {}
    # Use the planned workout name for intent. Descriptions commonly contain
    # words such as "easy" and "recovery" for non-target phases.
    planned_text = str(planned.get("name") or "").lower()
    activity_text = str(summary.get("name") or "").lower()

    for source, text in (
        ("planned_workout_name", planned_text),
        ("activity_text", activity_text),
    ):
        if not text.strip():
            continue
        for session_type, session_cfg in session_types:
            keywords = session_cfg["recognition"].get("keywords", [])
            matched_keyword = next(
                (keyword for keyword in keywords if _text_has_keyword(text, keyword)),
                None,
            )
            if matched_keyword:
                summary["session_type_source"] = source
                summary["session_type_evidence"] = {
                    "matched_keyword": matched_keyword,
                    "matched_text": text,
                }
                return session_type

    for session_type, session_cfg in session_types:
        recognition = session_cfg["recognition"]
        if recognition["mode"] != "intervals":
            continue
        evidence = _match_interval_signature(summary, session_type, recognition)
        if evidence:
            summary["session_type_source"] = "configured_interval_signature"
            summary["session_type_evidence"] = evidence
            _apply_inferred_session_policy(summary, session_type, evidence)
            return session_type

    if_value = summary.get("if_value")
    if if_value is not None:
        for session_type, session_cfg in session_types:
            if_bounds = session_cfg["recognition"].get("whole_activity_if")
            if if_bounds and if_bounds[0] <= if_value < if_bounds[1]:
                summary["session_type_source"] = "configured_whole_activity_if"
                summary["session_type_evidence"] = {
                    "if_value": if_value,
                    "configured_range": if_bounds,
                }
                return session_type

    summary["session_type_source"] = "unclassified_fallback"
    summary["session_type_evidence"] = {}
    return classification["default_session_type"]


def _text_has_keyword(text: str, keyword: str) -> bool:
    """Match a configured word or phrase without accidental substrings."""
    return bool(re.search(rf"(?<!\w){re.escape(keyword.lower())}(?!\w)", text))


def _apply_inferred_session_policy(
    summary: dict,
    session_type: str,
    evidence: dict,
) -> None:
    """Annotate observed matches without inventing planned phase roles."""
    policy = get_coaching_config()["inferred_session_policy"]
    matched_indices = set(evidence.get("interval_indices", []))
    observed_label = policy["matched_interval_label_template"].format(
        session_type=session_type
    )
    for index, interval in enumerate(summary.get("interval_details") or []):
        interval["planned_role"] = interval.get("planned_role")
        if index in matched_indices:
            interval["observed_role"] = observed_label
            interval["observed_role_source"] = "configured_interval_signature"
        else:
            interval["observed_role"] = policy["unmatched_interval_role"]
            interval["observed_role_source"] = None

    summary["analysis_constraints"] = {
        "planned_roles_verified": bool(
            summary.get("target_interval_membership_verified")
        ),
        "unmatched_interval_role": policy["unmatched_interval_role"],
        "whole_activity_context_only_metrics": policy[
            "whole_activity_context_only_metrics"
        ],
        "whole_activity_metrics_verdict_eligible": False,
        "missing_plan_must_not_reduce_verdict": policy[
            "missing_plan_must_not_reduce_verdict"
        ],
        "allow_well_for_supported_observed_execution": policy[
            "allow_well_for_supported_observed_execution"
        ],
    }


def _match_interval_signature(
    summary: dict,
    session_type: str,
    signature: dict,
) -> dict | None:
    """Return evidence when observed boundaries match a configured signature."""
    ftp = summary.get("ftp_W")
    power_bounds = signature.get("power_pct_ftp")
    if power_bounds and not ftp:
        return None

    candidates = []
    for index, interval in enumerate(summary.get("interval_details") or []):
        if interval.get("is_work") is False:
            continue
        duration = interval.get("duration_s") or 0
        power = interval.get("avg_power_W")
        cadence = interval.get("avg_cadence_rpm")
        duration_bounds = signature.get("duration_s")
        cadence_bounds = signature.get("cadence_rpm")
        if duration_bounds and not duration_bounds[0] <= duration <= duration_bounds[1]:
            continue
        if cadence_bounds and (
            cadence is None or not cadence_bounds[0] <= cadence <= cadence_bounds[1]
        ):
            continue
        power_pct = power / ftp * 100 if power and ftp else None
        if power_bounds and (
            power_pct is None or not power_bounds[0] <= power_pct <= power_bounds[1]
        ):
            continue
        candidates.append({
            "index": index,
            "duration_s": duration,
            "avg_power_W": power,
            "power_pct_ftp": round(power_pct, 1) if power_pct is not None else None,
            "avg_cadence_rpm": cadence,
        })

    max_duration_spread = signature.get("max_duration_spread_pct")
    max_power_spread = signature.get("max_power_spread_pct")
    candidate_groups = (
        [tuple(candidates)]
        if (
            len(candidates) >= signature["min_reps"]
            and max_duration_spread is None
            and max_power_spread is None
        )
        else combinations(candidates, signature["min_reps"])
    )
    for matched in candidate_groups:
        durations = [item["duration_s"] for item in matched]
        powers = [item["avg_power_W"] for item in matched if item["avg_power_W"]]
        duration_mean = sum(durations) / len(durations)
        duration_spread = (max(durations) - min(durations)) / duration_mean * 100
        power_spread = None
        if powers:
            power_mean = sum(powers) / len(powers)
            power_spread = (max(powers) - min(powers)) / power_mean * 100
        if max_duration_spread is not None and duration_spread > max_duration_spread:
            continue
        if (
            max_power_spread is not None
            and power_spread is not None
            and power_spread > max_power_spread
        ):
            continue
        return {
            "signature_session_type": session_type,
            "matched_interval_count": len(matched),
            "interval_indices": [item["index"] for item in matched],
            "durations_s": durations,
            "avg_power_W": [item["avg_power_W"] for item in matched],
            "power_pct_ftp": [item["power_pct_ftp"] for item in matched],
            "avg_cadence_rpm": [item["avg_cadence_rpm"] for item in matched],
            "duration_spread_pct": round(duration_spread, 1),
            "power_spread_pct": (
                round(power_spread, 1) if power_spread is not None else None
            ),
        }
    return None


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
