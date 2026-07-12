"""
Step 0 Verification Spike — Intervals.icu API access check.

Tests whether the coach's API key can read athlete data.
Run with: python spike_verify_api.py

Requires INTERVALS_API_KEY in environment or .env file.
"""

import os
import json
import sys
from datetime import date, timedelta

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env loaded manually or via shell


BASE_URL = "https://intervals.icu/api/v1"
API_KEY = os.environ.get("INTERVALS_API_KEY", "")

if not API_KEY:
    print("ERROR: INTERVALS_API_KEY not set. Copy .env.example to .env and fill it in.")
    sys.exit(1)

AUTH = ("API_KEY", API_KEY)


def call(method: str, path: str, **kwargs):
    """Make an authenticated request and return (status_code, body_or_error)."""
    url = f"{BASE_URL}{path}"
    try:
        resp = requests.request(method, url, auth=AUTH, timeout=15, **kwargs)
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        return resp.status_code, body
    except Exception as e:
        return None, str(e)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(label: str, status, body):
    ok = status == 200
    tag = "PASS" if ok else f"FAIL ({status})"
    print(f"\n[{tag}] {label}")
    if ok:
        if isinstance(body, dict):
            # Print a small excerpt
            keys = list(body.keys())[:8]
            print(f"       Keys: {keys}")
        elif isinstance(body, list):
            print(f"       List of {len(body)} items")
            if body:
                first = body[0]
                if isinstance(first, dict):
                    print(f"       First item keys: {list(first.keys())[:8]}")
    else:
        excerpt = str(body)[:300]
        print(f"       Response: {excerpt}")
    return ok, body


# ─────────────────────────────────────────────────────────────
# 1. Identify the coach and list coached athletes
# ─────────────────────────────────────────────────────────────
section("1 — Coach identity + athlete roster")

status, body = call("GET", "/athlete/0/profile")
ok, _ = check("GET /athlete/0/profile (coach identity)", status, body)

coach_id = None
if ok and isinstance(body, dict):
    coach_id = body.get("id")
    print(f"       Coach ID: {coach_id}, name: {body.get('name')}")

# Try the athlete-summary endpoint to list coached athletes
status, body = call("GET", "/athlete/0/athlete-summary.json")
ok_roster, roster_body = check("GET /athlete/0/athlete-summary.json (athlete roster)", status, body)

athletes = []
if ok_roster and isinstance(roster_body, list):
    athletes = roster_body
    print(f"       Found {len(athletes)} coached athletes:")
    for a in athletes:
        print(f"         - id={a.get('id')} name={a.get('name')}")

if not athletes:
    # Try alternative endpoint
    status2, body2 = call("GET", "/athlete/0/athletes")
    ok2, _ = check("GET /athlete/0/athletes (fallback roster)", status2, body2)
    if ok2 and isinstance(body2, list):
        athletes = body2

# Pick the first coached athlete that isn't the coach himself
athlete_id = None
for a in athletes:
    aid = a.get("id")
    if aid and str(aid) != str(coach_id):
        athlete_id = aid
        athlete_name = a.get("name", "unknown")
        break

if not athlete_id and athletes:
    # Use whichever is first
    athlete_id = athletes[0].get("id")
    athlete_name = athletes[0].get("name", "unknown")

if athlete_id:
    print(f"\n  >> Will use athlete id={athlete_id} ({athlete_name}) for further tests")
else:
    print("\n  >> No athlete ID found — subsequent tests will use athlete/0 (coach self)")
    athlete_id = 0
    athlete_name = "coach self"


# ─────────────────────────────────────────────────────────────
# 2. Fetch activities for that athlete
# ─────────────────────────────────────────────────────────────
section("2 — Activity list for athlete")

# Last 30 days
today = date.today()
since = (today - timedelta(days=30)).isoformat()
until = today.isoformat()

status, body = call(
    "GET",
    f"/athlete/{athlete_id}/activities",
    params={"oldest": since, "newest": until},
)
ok_acts, acts_body = check(
    f"GET /athlete/{athlete_id}/activities (last 30 days)", status, body
)

activity_id = None
activity_date = None
if ok_acts and isinstance(acts_body, list) and acts_body:
    # Find the most recent activity
    act = acts_body[0]
    activity_id = act.get("id")
    activity_date = (act.get("start_date_local") or "")[:10]
    print(f"\n  >> Will inspect activity id={activity_id} on {activity_date}")
    print(f"       type={act.get('type')}, name={act.get('name')}")
    print(f"       source hint: {act.get('source', 'n/a')}")
    if act.get("source") in ("STRAVA", "strava"):
        print("  !! WARNING: This activity was synced from Strava. May not have full streams.")
elif ok_acts and isinstance(acts_body, list):
    print("  >> No activities in last 30 days.")
else:
    print("  >> Could not retrieve activity list.")


# ─────────────────────────────────────────────────────────────
# 3. Fetch full activity detail + streams
# ─────────────────────────────────────────────────────────────
section("3 — Activity detail + streams")

if activity_id:
    status, body = call("GET", f"/athlete/{athlete_id}/activities/{activity_id}")
    ok_detail, detail_body = check(
        f"GET /athlete/{athlete_id}/activities/{activity_id} (detail)", status, body
    )
    if ok_detail and isinstance(detail_body, dict):
        source = detail_body.get("source", "unknown")
        print(f"       source={source}")
        if source in ("STRAVA", "strava"):
            print("  !! Activity is Strava-synced — streams may be absent or restricted.")

    # Streams (power, HR, cadence, altitude, etc.)
    status, body = call(
        "GET",
        f"/athlete/{athlete_id}/activities/{activity_id}/streams",
        params={"types": "time,watts,heartrate,cadence,altitude,distance,velocity_smooth"},
    )
    ok_streams, streams_body = check(
        f"GET /athlete/{athlete_id}/activities/{activity_id}/streams", status, body
    )
    if ok_streams:
        if isinstance(streams_body, dict):
            available = [k for k, v in streams_body.items() if v]
            print(f"       Available streams: {available}")
        elif isinstance(streams_body, list):
            print(f"       Returned list with {len(streams_body)} entries")
else:
    print("  >> Skipped — no activity ID available.")


# ─────────────────────────────────────────────────────────────
# 4. Fetch planned/calendar workout for that date
# ─────────────────────────────────────────────────────────────
section("4 — Planned workout (calendar)")

if activity_date:
    # Events endpoint covers calendar workouts
    status, body = call(
        "GET",
        f"/athlete/{athlete_id}/events",
        params={"oldest": activity_date, "newest": activity_date},
    )
    ok_events, events_body = check(
        f"GET /athlete/{athlete_id}/events on {activity_date}", status, body
    )
    if ok_events and isinstance(events_body, list):
        workouts = [e for e in events_body if e.get("type") == "Workout" or e.get("category") == "WORKOUT"]
        print(f"       Total events: {len(events_body)}, Workouts: {len(workouts)}")
        if workouts:
            w = workouts[0]
            print(f"       Workout name: {w.get('name')}, description length: {len(str(w.get('description', '')))}")
else:
    print("  >> Skipped — no activity date available.")


# ─────────────────────────────────────────────────────────────
# 5. Fetch wellness data for that date
# ─────────────────────────────────────────────────────────────
section("5 — Wellness (HRV, sleep, resting HR)")

if activity_date:
    status, body = call(
        "GET",
        f"/athlete/{athlete_id}/wellness/{activity_date}",
    )
    ok_wellness, wellness_body = check(
        f"GET /athlete/{athlete_id}/wellness/{activity_date}", status, body
    )
    if ok_wellness and isinstance(wellness_body, dict):
        relevant = {k: wellness_body.get(k) for k in ["hrvScore", "restingHR", "sleepSecs", "sleepScore", "ctl", "atl", "rampRate"]}
        print(f"       Relevant fields: {relevant}")
else:
    print("  >> Skipped — no activity date available.")

    # Try fetching wellness list for last 7 days anyway
    status, body = call(
        "GET",
        f"/athlete/{athlete_id}/wellness",
        params={"oldest": since, "newest": until},
    )
    ok_wl, _ = check(f"GET /athlete/{athlete_id}/wellness (list fallback)", status, body)


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
section("SUMMARY")
print("""
Review the PASS/FAIL lines above and report:
  1. Which of the 5 calls succeeded?
  2. Did wellness return data or 401/403?
  3. Was the test activity Strava-synced? Were streams available?
""")
