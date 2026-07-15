"""FastAPI application — routes, startup, shutdown."""

import logging
import threading
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import get_coaching_config_model, get_settings
from app.db.models import Base
from app.db.session import get_engine, get_session
from app.db import crud
from app.intervals import client as icu, poller

# Track activity IDs whose background analysis jobs failed
_failed_activity_ids: set[str] = set()

logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

templates = Jinja2Templates(directory="app/templates")

def _fmt_duration(seconds) -> str:
    if not seconds:
        return "—"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"

templates.env.filters["duration"] = _fmt_duration


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail startup with a clear validation error if the coach edited invalid YAML.
    get_coaching_config_model()
    Base.metadata.create_all(bind=get_engine())
    _sync_athletes_from_icu()
    poller.start_scheduler()
    yield
    poller.stop_scheduler()


app = FastAPI(title="Bike Coach AI", lifespan=lifespan)


def _sync_athletes_from_icu():
    db = get_session()
    try:
        athletes = icu.list_athletes()
        for a in athletes:
            crud.upsert_athlete(db, {"id": a["id"], "name": a["name"], "weight_kg": a.get("weight")})
        log.info("Synced %d athletes from Intervals.icu", len(athletes))
    except Exception:
        log.exception("Failed to sync athletes on startup")
    finally:
        db.close()


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Dashboard — analysis feed, auto-refreshes, athlete filter in URL ───────────

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, athlete_id: Optional[str] = None):
    db = get_session()
    try:
        athletes = crud.list_active_athletes(db)
        analyses = crud.list_analyses(db, athlete_id=athlete_id, limit=50)

        rows = []
        for analysis in analyses:
            activity = db.get(crud.Activity, analysis.activity_id)
            rows.append({"analysis": analysis, "activity": activity})

        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "athletes": athletes,
            "rows": rows,
            "selected_athlete_id": athlete_id,
        })
    finally:
        db.close()


# ── Athlete activity list — fetched live from Intervals.icu ───────────────────

@app.get("/athlete/{athlete_id}", response_class=HTMLResponse)
def athlete_activities(request: Request, athlete_id: str, days: int = 30):
    db = get_session()
    try:
        athlete = crud.get_athlete(db, athlete_id)
        if not athlete:
            raise HTTPException(404, "Athlete not found")

        athletes = crud.list_active_athletes(db)
        profile_ftp_W = icu.get_current_ride_ftp(athlete_id)

        since = date.today() - timedelta(days=days)
        raw_activities = icu.list_activities(athlete_id, since=since)

        # Fetch planned workouts for the whole range in one call
        # Build: {date_str: {event_id: name, ..., "__first__": name}}
        events = icu.get_events_range(athlete_id, since=since)
        planned_by_date: dict[str, dict] = {}
        for e in events:
            if not (e.get("type") == "Workout" or e.get("category") == "WORKOUT"):
                continue
            edate = str(e.get("start_date_local", "") or e.get("start_date", ""))[:10]
            if not edate:
                continue
            name = e.get("name") or e.get("description", "")
            if not name:
                continue
            if edate not in planned_by_date:
                planned_by_date[edate] = {"__first__": name}
            planned_by_date[edate][str(e.get("id", ""))] = name

        # For each activity, look up whether we already have an analysis
        activity_rows = []
        for act in raw_activities:
            act_id = str(act.get("id", ""))
            existing = crud.get_latest_analysis(db, act_id) if act_id else None
            act_date = str(act.get("start_date_local", ""))[:10]

            # Prescribed name: from calendar event (paired first, else first workout that day)
            prescribed_name = None
            day_events = planned_by_date.get(act_date, {})
            if day_events:
                paired_id = str(act.get("paired_event_id", "") or "")
                prescribed_name = day_events.get(paired_id) or day_events.get("__first__")

            # Fall back to DB session_summary_json if already analysed
            if not prescribed_name:
                act_db = db.get(crud.Activity, act_id) if act_id else None
                if act_db and act_db.session_summary_json:
                    planned = act_db.session_summary_json.get("planned_workout") or {}
                    prescribed_name = planned.get("name") or None

            activity_rows.append({
                "id": act_id,
                "date": act_date,
                "name": prescribed_name or act.get("name", ""),
                "type": act.get("type", ""),
                "duration_s": act.get("moving_time"),
                "tss": act.get("icu_training_load"),
                "avg_power": act.get("icu_average_watts"),
                "np": act.get("icu_weighted_avg_watts"),
                "source": act.get("source", ""),
                "analysis": existing,
            })

        return templates.TemplateResponse("athlete.html", {
            "request": request,
            "athlete": athlete,
            "profile_ftp_W": profile_ftp_W,
            "athletes": athletes,
            "activity_rows": activity_rows,
            "days": days,
        })
    finally:
        db.close()


# ── Analyse one activity → fire in background, return 202 immediately ─────────

@app.post("/athlete/{athlete_id}/analyse/{activity_id}")
def analyse_activity(athlete_id: str, activity_id: str):
    _failed_activity_ids.discard(activity_id)

    def _run():
        result = poller.process_activity(athlete_id, activity_id, force=True)
        if not result:
            log.warning("Analysis returned None for %s / %s", athlete_id, activity_id)
            _failed_activity_ids.add(activity_id)

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started"}, status_code=202)


# ── Re-analyse → redirect back to athlete page ────────────────────────────────

@app.post("/athlete/{athlete_id}/reanalyse/{activity_id}")
def reanalyse_activity(athlete_id: str, activity_id: str):
    _failed_activity_ids.discard(activity_id)

    def _run():
        result = poller.reanalyse_activity(activity_id)
        if not result:
            _failed_activity_ids.add(activity_id)

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started"}, status_code=202)


# ── Analysis status poll (used by JS to know when background job is done) ─────

@app.get("/api/analysis-status/{activity_id}")
def analysis_status(activity_id: str, after_id: Optional[int] = None):
    if activity_id in _failed_activity_ids:
        return JSONResponse({"done": False, "failed": True})
    db = get_session()
    try:
        analysis = crud.get_latest_analysis(db, activity_id)
        if after_id is not None:
            # Re-run: wait until a newer analysis exists
            done = analysis is not None and analysis.id != after_id
        else:
            done = analysis is not None
        return JSONResponse({"done": done})
    finally:
        db.close()


# ── Analysis detail ────────────────────────────────────────────────────────────

@app.get("/analysis/{analysis_id}", response_class=HTMLResponse)
def analysis_detail(request: Request, analysis_id: int):
    db = get_session()
    try:
        analysis = db.get(crud.Analysis, analysis_id)
        if not analysis:
            raise HTTPException(404, "Analysis not found")
        activity = db.get(crud.Activity, analysis.activity_id)
        athlete = crud.get_athlete(db, analysis.athlete_id) if analysis.athlete_id else None
        athletes = crud.list_active_athletes(db)
        all_analyses = crud.get_all_analyses_for_activity(db, analysis.activity_id)
        return templates.TemplateResponse("analysis_detail.html", {
            "request": request,
            "analysis": analysis,
            "activity": activity,
            "athlete": athlete,
            "athletes": athletes,
            "all_analyses": all_analyses,
        })
    finally:
        db.close()


# ── Athlete profile update ─────────────────────────────────────────────────────

@app.post("/athletes/{athlete_id}/profile")
def update_athlete_profile(
    athlete_id: str,
    lthr_bpm: Optional[float] = Form(None),
    max_hr_bpm: Optional[float] = Form(None),
    weight_kg: Optional[float] = Form(None),
    level: Optional[str] = Form(None),
    training_phase: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
):
    db = get_session()
    try:
        data = {"id": athlete_id}
        for field, val in [
            ("lthr_bpm", lthr_bpm), ("max_hr_bpm", max_hr_bpm),
            ("weight_kg", weight_kg), ("level", level),
            ("training_phase", training_phase), ("notes", notes),
        ]:
            if val is not None:
                data[field] = val
        crud.upsert_athlete(db, data)
        return RedirectResponse(f"/athlete/{athlete_id}", status_code=303)
    finally:
        db.close()


# ── Fitness & power data (JSON feeds for charts) ──────────────────────────────

@app.get("/api/athlete/{athlete_id}/fitness")
def athlete_fitness(athlete_id: str, days: int = 90):
    since = date.today() - timedelta(days=days)
    data = icu.get_wellness_range(athlete_id, since=since)
    return JSONResponse(data)


@app.get("/api/athlete/{athlete_id}/power-curve")
def athlete_power_curve(athlete_id: str, days: int = 90):
    since = date.today() - timedelta(days=days)
    data = icu.get_power_curve_range(athlete_id, since=since)
    data["ftp"] = icu.get_current_ride_ftp(athlete_id)
    data["ftp_source"] = "athlete_profile"
    return JSONResponse(data)


# ── Webhook stub ──────────────────────────────────────────────────────────────

@app.post("/webhook/intervals")
async def intervals_webhook(request: Request):
    payload = await request.json()
    athlete_id = payload.get("athlete_id")
    activity_id = payload.get("activity_id")
    if athlete_id and activity_id:
        result = poller.process_activity(str(athlete_id), str(activity_id))
        return JSONResponse({"status": "ok", "verdict": result.get("verdict") if result else None})
    return JSONResponse({"status": "ignored"})
