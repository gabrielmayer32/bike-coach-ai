from __future__ import annotations
"""
Calls the Claude API, requests structured JSON output, and logs cost.

System prompt is marked as ephemeral cache — identical on every call so
Anthropic's prompt caching reduces input token cost by ~90%.
"""

import json
import logging
from datetime import datetime

import anthropic

from app.config import get_settings
from app.ai.prompt_builder import build_system_prompt, build_user_message, build_athlete_context

log = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048

# Input token pricing (USD per million tokens) — update if pricing changes
PRICE_INPUT_PER_M = 3.00
PRICE_CACHE_WRITE_PER_M = 3.75
PRICE_CACHE_READ_PER_M = 0.30
PRICE_OUTPUT_PER_M = 15.00


def analyse_session(
    session_summary: dict,
    athlete_db_record: dict,
    similar_sessions: list[dict],
) -> dict:
    """
    Run a full coaching analysis for one session.

    Returns a dict with:
        verdict, key_observations, reasoning,
        escalate_to_coach, athlete_message,
        _meta (cost, tokens, timestamp)
    """
    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    wellness = session_summary.get("wellness", {})
    athlete_context = build_athlete_context(athlete_db_record, wellness)

    # Fill gaps in athlete context from the session summary (ICU values are authoritative)
    for field, summary_key in [
        ("ftp_W", "ftp_W"),
        ("lthr_bpm", "lthr_bpm"),
        ("max_hr_bpm", "athlete_max_hr"),
        ("weight_kg", "wellness.weight_kg"),
    ]:
        if not athlete_context.get(field):
            val = (
                session_summary.get("wellness", {}).get("weight_kg")
                if summary_key == "wellness.weight_kg"
                else session_summary.get(summary_key)
            )
            if val:
                athlete_context[field] = val

    system_prompt = build_system_prompt()
    user_message = build_user_message(session_summary, athlete_context, similar_sessions)

    log.info(
        "Calling Claude for activity=%s athlete=%s",
        session_summary.get("activity_id"),
        session_summary.get("athlete_id"),
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},  # prompt caching
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    raw_text = response.content[0].text.strip()

    # Strip markdown code fences if present
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        log.error("Claude returned invalid JSON: %s", raw_text[:500])
        raise ValueError(f"Claude returned invalid JSON: {exc}") from exc

    # ── Cost logging ──────────────────────────────────────────────────────────
    usage = response.usage
    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)
    cache_write = getattr(usage, "cache_creation_input_tokens", 0)
    cache_read = getattr(usage, "cache_read_input_tokens", 0)

    cost_usd = (
        (input_tokens / 1_000_000) * PRICE_INPUT_PER_M
        + (output_tokens / 1_000_000) * PRICE_OUTPUT_PER_M
        + (cache_write / 1_000_000) * PRICE_CACHE_WRITE_PER_M
        + (cache_read / 1_000_000) * PRICE_CACHE_READ_PER_M
    )

    meta = {
        "model": MODEL,
        "timestamp": datetime.utcnow().isoformat(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_write_tokens": cache_write,
        "cache_read_tokens": cache_read,
        "cost_usd": round(cost_usd, 6),
    }

    log.info(
        "Analysis complete: verdict=%s escalate=%s tokens_in=%d out=%d "
        "cache_read=%d cost=$%.4f",
        result.get("verdict"),
        result.get("escalate_to_coach", {}).get("flag"),
        input_tokens,
        output_tokens,
        cache_read,
        cost_usd,
    )

    result["_meta"] = meta
    result["_input_summary"] = {
        "session_summary": session_summary,
        "athlete_context": athlete_context,
        "similar_sessions_count": len(similar_sessions),
    }

    return result
