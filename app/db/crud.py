from __future__ import annotations
"""DB read/write helpers. All business logic stays out of here."""

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import Activity, Analysis, Athlete
from app.config import get_coaching_config


# ── Athletes ───────────────────────────────────────────────────────────────────

def upsert_athlete(db: Session, athlete_data: dict) -> Athlete:
    athlete = db.get(Athlete, athlete_data["id"])
    if not athlete:
        athlete = Athlete(id=athlete_data["id"])
        db.add(athlete)
    for key, val in athlete_data.items():
        if hasattr(athlete, key):
            setattr(athlete, key, val)
    athlete.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(athlete)
    return athlete


def get_athlete(db: Session, athlete_id: str) -> Athlete | None:
    return db.get(Athlete, athlete_id)


def list_active_athletes(db: Session) -> list[Athlete]:
    return db.query(Athlete).filter(Athlete.active == True).all()


def athlete_as_dict(athlete: Athlete) -> dict:
    return {
        "id": athlete.id,
        "name": athlete.name,
        "level": athlete.level,
        "training_phase": athlete.training_phase,
        "ftp_W": athlete.ftp_W,
        "lthr_bpm": athlete.lthr_bpm,
        "max_hr_bpm": athlete.max_hr_bpm,
        "weight_kg": athlete.weight_kg,
        "training_age_years": athlete.training_age_years,
        "notes": athlete.notes,
    }


# ── Activities ─────────────────────────────────────────────────────────────────

def activity_exists(db: Session, activity_id: str) -> bool:
    return db.query(Activity).filter(Activity.id == activity_id).count() > 0


def store_activity(db: Session, session_summary: dict, session_type: str) -> Activity:
    activity = db.get(Activity, session_summary["activity_id"])
    if not activity:
        activity = Activity(id=session_summary["activity_id"])
        db.add(activity)

    activity.athlete_id = session_summary["athlete_id"]
    activity.date = session_summary["date"]
    activity.name = session_summary.get("name", "")
    activity.activity_type = session_summary.get("type", "")
    activity.session_type = session_type
    activity.source = session_summary.get("source", "")
    activity.is_indoor = session_summary.get("is_indoor", False)
    activity.avg_power_W = session_summary.get("avg_power_W")
    activity.np_W = session_summary.get("np_W")
    activity.ftp_W = session_summary.get("ftp_W")
    activity.if_value = session_summary.get("if_value")
    activity.tss = session_summary.get("tss")
    activity.vi = session_summary.get("vi")
    activity.decoupling_pct = session_summary.get("decoupling_pct")
    activity.rep_fade_pct = session_summary.get("rep_fade_pct")
    activity.avg_hr_bpm = session_summary.get("avg_hr_bpm")
    activity.avg_cadence_rpm = session_summary.get("avg_cadence_rpm")
    activity.duration_s = session_summary.get("duration_s")
    activity.kj = session_summary.get("kj")
    activity.rpe = session_summary.get("rpe")
    activity.temp_c = session_summary.get("temp_c")
    activity.session_summary_json = session_summary
    activity.fetched_at = datetime.utcnow()

    db.commit()
    db.refresh(activity)
    return activity


def get_similar_sessions(
    db: Session,
    athlete_id: str,
    session_type: str,
    exclude_activity_id: str,
    limit: int | None = None,
) -> list[dict]:
    """
    Return the last N analysed sessions of the same type for an athlete.
    Used to build the historical comparison context for the AI.
    """
    history_cfg = get_coaching_config()["historical_comparison"]
    if not history_cfg["enabled"]:
        return []
    n = history_cfg["num_sessions"] if limit is None else limit
    if n <= 0:
        return []

    rows = (
        db.query(Activity, Analysis)
        .join(Analysis, Analysis.activity_id == Activity.id)
        .filter(
            Activity.athlete_id == athlete_id,
            Activity.session_type == session_type,
            Activity.id != exclude_activity_id,
        )
        .order_by(Activity.date.desc())
        .limit(n)
        .all()
    )

    result = []
    for act, analysis in rows:
        item = {"date": act.date}
        summary = act.session_summary_json or {}
        for field in history_cfg["fields_to_compare"]:
            if field == "verdict":
                item[field] = analysis.verdict
            elif hasattr(act, field):
                item[field] = getattr(act, field)
            else:
                item[field] = summary.get(field)
        result.append(item)
    return result


# ── Analyses ───────────────────────────────────────────────────────────────────

def store_analysis(db: Session, activity_id: str, athlete_id: str, ai_result: dict) -> Analysis:
    meta = ai_result.get("_meta", {})
    escalate = ai_result.get("escalate_to_coach", {})

    analysis = Analysis(
        activity_id=activity_id,
        athlete_id=athlete_id,
        verdict=ai_result.get("verdict"),
        key_observations=ai_result.get("key_observations", []),
        reasoning=ai_result.get("reasoning", ""),
        athlete_message=ai_result.get("athlete_message", ""),
        escalate_flag=escalate.get("flag", False),
        escalate_reason=escalate.get("reason", ""),
        model=meta.get("model"),
        input_tokens=meta.get("input_tokens"),
        output_tokens=meta.get("output_tokens"),
        cache_write_tokens=meta.get("cache_write_tokens"),
        cache_read_tokens=meta.get("cache_read_tokens"),
        cost_usd=meta.get("cost_usd"),
        input_summary_json=ai_result.get("_input_summary"),
        raw_output_json={k: v for k, v in ai_result.items() if not k.startswith("_")},
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis


def get_latest_analysis(db: Session, activity_id: str) -> Analysis | None:
    return (
        db.query(Analysis)
        .filter(Analysis.activity_id == activity_id)
        .order_by(Analysis.created_at.desc())
        .first()
    )


def list_analyses(db: Session, athlete_id: str | None = None, limit: int = 50) -> list[Analysis]:
    q = db.query(Analysis)
    if athlete_id:
        q = q.filter(Analysis.athlete_id == athlete_id)
    return q.order_by(Analysis.created_at.desc()).limit(limit).all()


def get_all_analyses_for_activity(db: Session, activity_id: str) -> list[Analysis]:
    """All historical analyses for an activity — supports reanalysis comparison."""
    return (
        db.query(Analysis)
        .filter(Analysis.activity_id == activity_id)
        .order_by(Analysis.created_at.asc())
        .all()
    )
