from __future__ import annotations
"""
Builds the system prompt and user message for each analysis call.

The system prompt is assembled from coaching_config_editable.yaml and stays
identical until the coach saves a change, enabling Anthropic prompt caching.

The user message contains only the computed session summary and athlete context
for this one analysis — no raw data, no streams.
"""

import json
from typing import Any

from app.config import get_coaching_config


def build_system_prompt() -> str:
    """
    Assemble the full system prompt from the live validated coaching config.
    """
    cfg = get_coaching_config()
    persona = cfg["persona"]
    voice = cfg["voice"]
    tolerances = cfg["tolerances"]
    session_types = cfg["session_types"]
    verdict = cfg["verdict"]
    context_mods = cfg["context_modifiers"]
    escalation = cfg["escalation"]
    output_spec = cfg["output_spec"]
    examples = cfg["examples"]

    # ── Persona & philosophy ──────────────────────────────────────────────────
    parts = [
        f"You are the coaching analysis assistant for {cfg['coach']['name']}.",
        f"Write athlete feedback in language code: {cfg['coach']['language']}.",
        "",
        "## Coach Profile",
        persona["bio"],
        "",
        "## Coaching Philosophy",
        persona["philosophy"],
        "",
        "## What makes this coaching different",
        persona["differentiator"],
    ]

    # ── Voice & feedback style ────────────────────────────────────────────────
    parts += [
        "",
        "## Voice and Feedback Style",
        voice["tone"],
        f"Start naturally with the configured greeting: {voice['greeting']}.",
        f"**LENGTH: EXACTLY {voice['feedback_length_sentences']} sentences. No more. Count before you finish.**",
        "",
        f"### Feedback structure ({len(voice['structure'])} steps):",
    ]
    for step in voice["structure"]:
        parts.append(f"{step['step']}. **{step['label'].upper()}**: {step['instruction']}")

    parts += [
        "",
        "### Phrases to use naturally (not all at once):",
        ", ".join(f'"{p}"' for p in voice["phrases_to_use_naturally"]),
        "",
        "### Words/phrases to NEVER use:",
        ", ".join(f'"{w}"' for w in voice["words_to_avoid"]),
        "",
        "### Hard rules:",
    ]
    for rule in voice["never_do"]:
        parts.append(f"- {rule}")

    # ── Verdict criteria ──────────────────────────────────────────────────────
    parts += [
        "",
        "## Verdict Criteria",
        f"- **well**: {verdict['well']['description']}",
        f"- **okay**: {verdict['okay']['description']}",
        f"- **poor**: {verdict['poor']['description']}",
    ]

    # ── Metric tolerances ─────────────────────────────────────────────────────
    parts += [
        "",
        "## Metric Tolerances",
        "Use these inclusive boundaries to decide well/okay/poor. Context can override.",
    ]
    for metric, thresholds in tolerances.items():
        parts.append(_format_tolerance(metric, thresholds))

    # ── Session type reference ────────────────────────────────────────────────
    parts += ["", "## Session Type Reference"]
    for key, st in session_types.items():
        parts.append(f"\n### {st['label']} (`{key}`)")
        parts.append(f"Purpose: {st['purpose'].strip()}")
        if st.get("target_zone") is not None:
            parts.append(f"Target zone: Z{st['target_zone']}")
        if st.get("target_cadence_rpm") is not None:
            parts.append(f"Target cadence: {st['target_cadence_rpm']} rpm")
        if st.get("key_metrics"):
            parts.append("Key metrics: " + ", ".join(st["key_metrics"]))
        if st.get("pacing_preference"):
            parts.append(f"Pacing preference: {st['pacing_preference']}")
        parts.append(f"Well executed: {st['well_executed'].strip()}")
        if st.get("red_flags"):
            parts.append("Red flags: " + "; ".join(st["red_flags"]))

    # ── Context modifiers ─────────────────────────────────────────────────────
    parts += ["", "## Context Modifiers (when to relax tolerances)"]
    for key, mod in context_mods.items():
        parts.append(f"- **{mod['trigger']}**: {mod['effect'].strip()}")

    # ── Escalation ────────────────────────────────────────────────────────────
    parts += ["", "## When to Escalate to Coach (set escalate_to_coach.flag = true)"]
    for rule in escalation["escalate_when"]:
        parts.append(f"- {rule}")

    # ── Output specification ──────────────────────────────────────────────────
    parts += [
        "",
        "## Output Specification",
        output_spec["instruction"].strip(),
        "",
        "Respond with ONLY valid JSON in this exact shape:",
        json.dumps(_output_example(output_spec["fields"]), indent=2),
        "Configured field schema:",
        json.dumps(output_spec["fields"], indent=2),
    ]

    # ── Few-shot examples ─────────────────────────────────────────────────────
    parts += ["", "## Worked Examples"]
    for ex in examples:
        parts.append(f"\n### {ex['label']}")
        if ex.get("context"):
            parts.append("Context:")
            parts.append(json.dumps(ex["context"], indent=2))
        parts.append("Session summary:")
        parts.append(json.dumps(ex.get("computed_summary", ex.get("session_summary", {})), indent=2))
        parts.append(f"Verdict: {ex['verdict']}")
        parts.append("Athlete message:")
        parts.append(ex["athlete_message"].strip())

    return "\n".join(parts)


def _format_tolerance(metric: str, thresholds: dict) -> str:
    """Render config thresholds with explicit, non-overlapping boundaries."""
    well = thresholds.get("well", thresholds.get("well_pct"))
    okay = thresholds.get("okay", thresholds.get("okay_pct"))
    target = thresholds.get("target")
    higher_is_better = thresholds.get("higher_is_better", False)
    target_text = f", target={target}" if target is not None else ""
    if well is None or okay is None:
        return f"- {metric}: {thresholds}"
    if higher_is_better:
        rule = f"well >= {well}; okay >= {okay} and < {well}; poor < {okay}"
    else:
        rule = f"well <= {well}; okay > {well} and <= {okay}; poor > {okay}"
    return f"- {metric}: {rule}{target_text}"


def _output_example(fields: dict) -> dict:
    """Build a concrete response example directly from the configured schema."""
    example = {}
    for name, spec in fields.items():
        field_type = spec.get("type")
        if spec.get("enum"):
            example[name] = " | ".join(spec["enum"])
        elif field_type == "list":
            example[name] = [spec.get("description", "string")]
        elif field_type == "object":
            example[name] = {
                child: False
                if (child_spec.get("type") if isinstance(child_spec, dict) else child_spec) == "boolean"
                else ""
                for child, child_spec in spec.get("fields", {}).items()
            }
        else:
            example[name] = spec.get("description", "string")
    return example


def build_user_message(
    session_summary: dict,
    athlete_context: dict,
    similar_sessions: list[dict],
) -> str:
    """
    Build the per-analysis user message. Contains only computed facts.
    No raw streams, no data the AI needs to re-derive.
    """
    parts = ["## Session to Analyse", json.dumps(session_summary, indent=2, default=str)]

    parts += ["", "## Athlete Context", json.dumps(athlete_context, indent=2, default=str)]

    if similar_sessions:
        parts += [
            "",
            f"## Last {len(similar_sessions)} Similar Sessions (for trend context)",
            json.dumps(similar_sessions, indent=2, default=str),
        ]

    parts += [
        "",
        "Analyse this session. Reason through the metrics first, then write the athlete message.",
        "Respond with valid JSON only.",
    ]

    return "\n".join(parts)


def build_athlete_context(
    athlete_db_record: dict,
    wellness_summary: dict,
) -> dict:
    """Combine stored athlete profile with today's wellness into a single context dict."""
    return {
        "name": athlete_db_record.get("name", ""),
        "level": athlete_db_record.get("level", ""),
        "training_phase": athlete_db_record.get("training_phase", ""),
        # FTP is injected from the Intervals activity summary by analyse_session.
        "ftp_W": None,
        "ftp_source": "missing",
        "lthr_bpm": athlete_db_record.get("lthr_bpm"),
        "max_hr_bpm": athlete_db_record.get("max_hr_bpm"),
        "weight_kg": athlete_db_record.get("weight_kg"),
        "training_age_years": athlete_db_record.get("training_age_years"),
        "notes": athlete_db_record.get("notes", ""),
        "wellness_today": wellness_summary,
    }
