from __future__ import annotations
"""
Builds the system prompt and user message for each analysis call.

The system prompt is assembled once from coaching_config.yaml and is identical
on every call — enabling Anthropic prompt caching to cut input token cost ~90%.

The user message contains only the computed session summary and athlete context
for this one analysis — no raw data, no streams.
"""

import json
from typing import Any

from app.config import get_coaching_config


def build_system_prompt() -> str:
    """
    Assemble the full system prompt from coaching_config.yaml.
    This is stable across calls → cached by Anthropic.
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
        f"Target length: {voice['feedback_length_sentences']} sentences.",
        "",
        "### Feedback structure (follow in order):",
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
        "Use these to decide well/okay/poor. Context can override — see modifiers below.",
        "",
        f"Power compliance vs target: well=±{tolerances['power_compliance_pct']['well']}%,"
        f" okay=±{tolerances['power_compliance_pct']['okay']}%, poor=>±{tolerances['power_compliance_pct']['okay']}%",
        f"Rep-to-rep fade: well=<{tolerances['rep_fade_pct']['well']}%,"
        f" okay={tolerances['rep_fade_pct']['well']}–{tolerances['rep_fade_pct']['okay']}%,"
        f" poor=>{tolerances['rep_fade_pct']['okay']}%",
        f"Sprint fade: well=<{tolerances['sprint_fade_pct']['well']}%, poor=>{tolerances['sprint_fade_pct']['okay']}%",
        f"HR/Power decoupling: well=<{tolerances['decoupling_pct']['well']}%,"
        f" okay={tolerances['decoupling_pct']['well']}–{tolerances['decoupling_pct']['okay']}%,"
        f" poor=>{tolerances['decoupling_pct']['okay']}%",
        f"Time in target zone (Z2): well=>{tolerances['time_in_zone_pct']['well']}%,"
        f" okay={tolerances['time_in_zone_pct']['okay']}–{tolerances['time_in_zone_pct']['well']}%,"
        f" poor=<{tolerances['time_in_zone_pct']['okay']}%",
        f"Variability index (single interval): well=<{tolerances['variability_index']['well']},"
        f" poor=>{tolerances['variability_index']['okay']}",
        f"VO2max time at intensity: well=>{tolerances['vo2_time_at_intensity_pct']['well']}%,"
        f" poor=<{tolerances['vo2_time_at_intensity_pct']['okay']}%",
        f"Sprint vs 90-day best: well=>{tolerances['sprint_vs_90d_best_pct']['well']}%,"
        f" poor=<{tolerances['sprint_vs_90d_best_pct']['okay']}%",
        f"RPE mismatch: well=±{tolerances['rpe_mismatch_points']['well']} pt,"
        f" poor=±{tolerances['rpe_mismatch_points']['okay']}+ pts",
        f"Within-rep fade (first vs second half): well=<{tolerances['within_interval_pacing']['well_pct']}% drop,"
        f" poor=>{tolerances['within_interval_pacing']['okay_pct']}% drop",
    ]

    # ── Session type reference ────────────────────────────────────────────────
    parts += ["", "## Session Type Reference"]
    for key, st in session_types.items():
        parts.append(f"\n### {st['label']}")
        parts.append(f"Purpose: {st['purpose'].strip()}")
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
        json.dumps(
            {
                "verdict": "well | okay | poor",
                "key_observations": ["2–4 bullet strings"],
                "reasoning": "Internal reasoning before writing the athlete message",
                "escalate_to_coach": {"flag": False, "reason": ""},
                "athlete_message": "The feedback text in Alexandre's voice",
            },
            indent=2,
        ),
    ]

    # ── Few-shot examples ─────────────────────────────────────────────────────
    parts += ["", "## Worked Examples"]
    for ex in examples:
        parts.append(f"\n### {ex['label']}")
        parts.append("Session summary:")
        parts.append(json.dumps(ex.get("computed_summary", ex.get("session_summary", {})), indent=2))
        parts.append(f"Verdict: {ex['verdict']}")
        parts.append("Athlete message:")
        parts.append(ex["athlete_message"].strip())

    return "\n".join(parts)


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
        "ftp_W": athlete_db_record.get("ftp_W"),
        "lthr_bpm": athlete_db_record.get("lthr_bpm"),
        "max_hr_bpm": athlete_db_record.get("max_hr_bpm"),
        "weight_kg": athlete_db_record.get("weight_kg"),
        "training_age_years": athlete_db_record.get("training_age_years"),
        "notes": athlete_db_record.get("notes", ""),
        "wellness_today": wellness_summary,
    }
