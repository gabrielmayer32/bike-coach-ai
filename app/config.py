from __future__ import annotations
"""App-wide settings and coaching config loader."""

import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    intervals_api_key: str
    anthropic_api_key: str
    database_url: str = "sqlite:///./bike_coach.db"
    poll_interval_seconds: int = 300
    log_level: str = "INFO"

@lru_cache
def get_settings() -> Settings:
    return Settings()


class CoachConfig(BaseModel):
    name: str
    language: str = "en"


class PersonaConfig(BaseModel):
    bio: str
    philosophy: str
    differentiator: str


class VoiceStep(BaseModel):
    step: int
    label: str
    instruction: str


class VoiceConfig(BaseModel):
    greeting: str = "Hi"
    tone: str
    feedback_length_sentences: int = Field(ge=1, le=10)
    structure: list[VoiceStep] = Field(min_length=1)
    phrases_to_use_naturally: list[str]
    words_to_avoid: list[str]
    never_do: list[str]


class ZoneDefinition(BaseModel):
    zone: int = Field(ge=1)
    name: str
    power_pct_ftp: tuple[float, float]
    hr_pct_lthr: Optional[tuple[float, float]] = None

    @model_validator(mode="after")
    def validate_bounds(self) -> "ZoneDefinition":
        if self.power_pct_ftp[0] > self.power_pct_ftp[1]:
            raise ValueError("power_pct_ftp lower bound must not exceed upper bound")
        if self.hr_pct_lthr and self.hr_pct_lthr[0] > self.hr_pct_lthr[1]:
            raise ValueError("hr_pct_lthr lower bound must not exceed upper bound")
        return self


class ZonesConfig(BaseModel):
    default_model: str
    coggan_7: list[ZoneDefinition] = Field(min_length=1)


class SessionTypeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str
    purpose: str
    target_zone: Optional[int] = None
    key_metrics: list[str] = Field(default_factory=list)
    well_executed: str
    pacing_preference: Optional[str] = None
    target_cadence_rpm: Optional[float] = None
    red_flags: list[str] = Field(default_factory=list)


class VerdictItem(BaseModel):
    label: Optional[str] = None
    description: str


class ContextModifier(BaseModel):
    trigger: str
    effect: str


class EscalationConfig(BaseModel):
    escalate_when: list[str]
    inactive_longitudinal_rules: list[str] = Field(default_factory=list)


class HistoricalComparisonConfig(BaseModel):
    enabled: bool = True
    num_sessions: int = Field(default=3, ge=0, le=20)
    match_on: list[Literal["session_type"]] = Field(default_factory=lambda: ["session_type"])
    fields_to_compare: list[str]


class OutputSpecConfig(BaseModel):
    format: Literal["json"] = "json"
    fields: dict[str, Any]
    instruction: str


class ClassificationKeywordRule(BaseModel):
    session_type: str
    keywords: list[str] = Field(min_length=1)


class ClassificationIfRule(BaseModel):
    session_type: str
    below: Optional[float] = None


class ClassificationConfig(BaseModel):
    keyword_priority: list[ClassificationKeywordRule]
    low_cadence_session_type: str = "torque"
    low_cadence_min_rpm: float = 40
    low_cadence_max_rpm: float = 65
    low_cadence_min_if: float = 0.7
    if_fallback: list[ClassificationIfRule]
    default_session_type: str


class CoachingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coach: CoachConfig
    persona: PersonaConfig
    voice: VoiceConfig
    zones: ZonesConfig
    tolerances: dict[str, dict[str, Any]]
    classification: ClassificationConfig
    session_types: dict[str, SessionTypeConfig]
    verdict: dict[str, VerdictItem]
    context_modifiers: dict[str, ContextModifier]
    escalation: EscalationConfig
    historical_comparison: HistoricalComparisonConfig
    output_spec: OutputSpecConfig
    examples: list[dict[str, Any]]

    @model_validator(mode="after")
    def validate_references(self) -> "CoachingConfig":
        known = set(self.session_types)
        referenced = {
            rule.session_type for rule in self.classification.keyword_priority
        }
        referenced.update(rule.session_type for rule in self.classification.if_fallback)
        referenced.add(self.classification.low_cadence_session_type)
        referenced.add(self.classification.default_session_type)
        unknown = referenced - known
        if unknown:
            raise ValueError(f"classification references unknown session types: {sorted(unknown)}")
        if self.zones.default_model != "coggan_7":
            raise ValueError("only the coggan_7 zone model is currently supported")
        if self.voice.feedback_length_sentences != len(self.voice.structure):
            raise ValueError(
                "voice.feedback_length_sentences must equal the number of structure steps"
            )
        zone_bounds = [zone.power_pct_ftp for zone in self.zones.coggan_7]
        for previous, current in zip(zone_bounds, zone_bounds[1:]):
            if current[0] != previous[1]:
                raise ValueError("power zone boundaries must be continuous")
        required_output_fields = {
            "verdict", "key_observations", "reasoning",
            "escalate_to_coach", "athlete_message",
        }
        if set(self.output_spec.fields) != required_output_fields:
            raise ValueError(
                "output_spec.fields must match the AnalysisResult interface: "
                f"{sorted(required_output_fields)}"
            )
        for metric, thresholds in self.tolerances.items():
            if "higher_is_better" not in thresholds:
                raise ValueError(f"tolerance {metric} must define higher_is_better")
        allowed_history_fields = {
            "avg_power_W", "np_W", "rep_fade_pct", "decoupling_pct",
            "time_in_target_zone_pct", "tss", "verdict", "if_value", "vi",
            "rpe", "avg_hr_bpm", "avg_cadence_rpm", "ftp_W", "duration_s", "kj",
        }
        unknown_history_fields = (
            set(self.historical_comparison.fields_to_compare) - allowed_history_fields
        )
        if unknown_history_fields:
            raise ValueError(
                "historical_comparison contains unsupported fields: "
                f"{sorted(unknown_history_fields)}"
            )
        return self


COACHING_CONFIG_PATH = Path(__file__).parent.parent / "coaching_config_editable.yaml"
_coaching_config_lock = threading.Lock()
_coaching_config_cache: tuple[int, CoachingConfig] | None = None


def get_coaching_config_model() -> CoachingConfig:
    """Return validated config, reloading automatically when the YAML changes."""
    global _coaching_config_cache
    modified_ns = COACHING_CONFIG_PATH.stat().st_mtime_ns
    with _coaching_config_lock:
        if _coaching_config_cache and _coaching_config_cache[0] == modified_ns:
            return _coaching_config_cache[1]
        with COACHING_CONFIG_PATH.open(encoding="utf-8") as config_file:
            raw = yaml.safe_load(config_file)
        config = CoachingConfig.model_validate(raw)
        _coaching_config_cache = (modified_ns, config)
        return config


def get_coaching_config() -> dict[str, Any]:
    """Compatibility mapping for code that consumes config as dictionaries."""
    return get_coaching_config_model().model_dump()


def clear_coaching_config_cache() -> None:
    """Test/support hook; normal runtime reloads are modification-time based."""
    global _coaching_config_cache
    with _coaching_config_lock:
        _coaching_config_cache = None
