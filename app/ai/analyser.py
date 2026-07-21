from __future__ import annotations
"""
Calls the Claude API, requests structured JSON output, and logs cost.

System prompt is marked as ephemeral cache — identical on every call so
Anthropic's prompt caching reduces input token cost by ~90%.
"""

import json
import logging
import re
from datetime import datetime
from typing import Literal

import anthropic
from pydantic import BaseModel, Field, ValidationError

from app.config import get_coaching_config, get_settings
from app.ai.prompt_builder import build_system_prompt, build_user_message, build_athlete_context

log = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048

# Input token pricing (USD per million tokens) — update if pricing changes
PRICE_INPUT_PER_M = 3.00
PRICE_CACHE_WRITE_PER_M = 3.75
PRICE_CACHE_READ_PER_M = 0.30
PRICE_OUTPUT_PER_M = 15.00


class EscalationResult(BaseModel):
    flag: bool
    reason: str = ""


class AnalysisResult(BaseModel):
    verdict: Literal["well", "okay", "poor"]
    key_observations: list[str] = Field(min_length=2, max_length=4)
    reasoning: str
    escalate_to_coach: EscalationResult
    athlete_message: str


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

    # Fill context from the session summary. FTP is always overwritten because
    # the local athlete value is display-only and Intervals is authoritative.
    athlete_context["ftp_W"] = session_summary.get("ftp_W")
    athlete_context["ftp_source"] = session_summary.get("ftp_source", "missing")
    for field, summary_key in [
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

    messages = [{"role": "user", "content": user_message}]
    responses = []
    result = None
    validation_errors: list[str] = []
    for attempt in range(2):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=messages,
        )
        responses.append(response)
        raw_text = _strip_code_fence(response.content[0].text.strip())
        try:
            parsed = json.loads(raw_text)
            validated = AnalysisResult.model_validate(parsed)
            _validate_athlete_message(validated.athlete_message)
            _validate_analysis_grounding(validated, session_summary)
            result = validated.model_dump()
            break
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            validation_errors = _validation_messages(exc)
            if attempt == 1:
                log.error("Claude output failed validation: %s", validation_errors)
                raise ValueError(
                    "Claude output failed validation after one corrective retry: "
                    + "; ".join(validation_errors)
                ) from exc
            messages.extend([
                {"role": "assistant", "content": raw_text},
                {
                    "role": "user",
                    "content": (
                        "Your response failed validation. Correct every issue and return "
                        "only the complete JSON object. Issues: "
                        + "; ".join(validation_errors)
                    ),
                },
            ])

    assert result is not None

    # ── Cost logging ──────────────────────────────────────────────────────────
    input_tokens = sum(getattr(item.usage, "input_tokens", 0) for item in responses)
    output_tokens = sum(getattr(item.usage, "output_tokens", 0) for item in responses)
    cache_write = sum(getattr(item.usage, "cache_creation_input_tokens", 0) for item in responses)
    cache_read = sum(getattr(item.usage, "cache_read_input_tokens", 0) for item in responses)

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


def _strip_code_fence(raw_text: str) -> str:
    if not raw_text.startswith("```"):
        return raw_text
    fenced = raw_text.split("```", 2)[1]
    if fenced.startswith("json"):
        fenced = fenced[4:]
    return fenced.strip()


def _validate_athlete_message(message: str) -> None:
    cfg = get_coaching_config()["voice"]
    errors = []
    sentence_count = len(re.findall(r"[.!?]+(?:\s|$)", message.strip()))
    expected = cfg["feedback_length_sentences"]
    if sentence_count != expected:
        errors.append(f"athlete_message must contain exactly {expected} sentences, got {sentence_count}")
    if any(dash in message for dash in ("-", "–", "—")):
        errors.append("athlete_message contains a forbidden dash")
    lowered = message.lower()
    for word in cfg["words_to_avoid"]:
        if re.search(rf"\b{re.escape(word.lower())}\b", lowered):
            errors.append(f"athlete_message contains forbidden word: {word}")
    for phrase in ("this tells us", "this tells me", "which means", "because aerobic"):
        if phrase in lowered:
            errors.append(f"athlete_message contains forbidden explanation: {phrase}")
    if errors:
        raise ValueError("; ".join(errors))


def _validate_analysis_grounding(
    result: AnalysisResult,
    session_summary: dict,
) -> None:
    """Reject invented phase roles and unsupported verdict penalties."""
    if session_summary.get("target_interval_membership_verified"):
        return

    cfg = get_coaching_config()
    policy = cfg["inferred_session_policy"]
    session_type = session_summary.get("session_type")
    recognition_mode = (
        cfg["session_types"].get(session_type, {}).get("recognition", {}).get("mode")
        if session_type
        else None
    )
    if not (
        session_summary.get("analysis_constraints")
        or session_summary.get("session_type_source") == "configured_interval_signature"
        or recognition_mode in {"intervals", "text_only"}
    ):
        return
    text = " ".join([
        *result.key_observations,
        result.reasoning,
        result.athlete_message,
    ])
    sentences = re.split(r"(?<=[.!?])\s+", text)
    errors: list[str] = []

    for label in policy["forbidden_unverified_phase_labels"]:
        phase_pattern = re.compile(
            rf"\b{re.escape(label.lower())}(?:/[\w-]+)?(?:\s+\w+){{0,2}}\s+"
            rf"(?:lap|interval|phase|section)s?\b"
        )
        if phase_pattern.search(text.lower()):
            errors.append(
                f"unverified interval phase label is forbidden: {label}"
            )

    negation_qualifiers = (
        "cannot be confirmed",
        "not confirmed",
        "not verified",
        "unverified",
        "unknown",
        "not linked",
    )
    for sentence in sentences:
        lowered = sentence.lower()
        for term in policy["unqualified_work_terms"]:
            if term.lower() in lowered and not any(
                qualifier in lowered for qualifier in negation_qualifiers
            ):
                errors.append(f"unqualified planned-role term is forbidden: {term}")

        verdict_causal = (
            ("verdict" in lowered or "okay rather than well" in lowered)
            and any(word in lowered for word in ("because", "reason", "rather than"))
        )
        if verdict_causal and any(
            phrase in lowered
            for phrase in (
                "plan is not linked",
                "planned workout is not linked",
                "target membership is unverified",
                "target membership is not verified",
            )
        ):
            errors.append("missing planned-workout context must not reduce the verdict")

        metric_mentioned = any(
            (
                metric == "vi" and re.search(r"\bvi\b|variability index", lowered)
            )
            or (metric == "decoupling_pct" and "decoupling" in lowered)
            or (metric == "time_in_target_zone_pct" and "time in" in lowered and "zone" in lowered)
            for metric in policy["whole_activity_context_only_metrics"]
        )
        contextualised = any(
            phrase in lowered
            for phrase in ("context only", "does not affect", "not used", "must not lower")
        )
        if verdict_causal and metric_mentioned and not contextualised:
            errors.append(
                "whole-activity context-only metrics must not reduce the verdict"
            )

    if errors:
        raise ValueError("; ".join(dict.fromkeys(errors)))


def _validation_messages(exc: Exception) -> list[str]:
    if isinstance(exc, ValidationError):
        return [
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        ]
    return [str(exc)]
