from __future__ import annotations

import os
import json
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app import config
from app.ai import analyser
from app.ai.analyser import _validate_athlete_message
from app.ai.prompt_builder import build_system_prompt
from app.db import crud
from app.db.models import Activity, Analysis, Athlete, Base
from app.intervals import client as icu
from app.intervals.poller import _classify_session
from app.metrics import compute
from app.metrics.session_summary import (
    _extract_interval_list,
    build_session_summary,
    enrich_session_summary,
)


def test_editable_config_is_the_only_canonical_file():
    assert config.COACHING_CONFIG_PATH.name == "coaching_config_editable.yaml"
    assert config.COACHING_CONFIG_PATH.exists()
    assert not config.COACHING_CONFIG_PATH.with_name("coaching_config.yaml").exists()


def test_config_reloads_when_file_changes(tmp_path, monkeypatch):
    raw = yaml.safe_load(config.COACHING_CONFIG_PATH.read_text())
    path = tmp_path / "coaching_config_editable.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    monkeypatch.setattr(config, "COACHING_CONFIG_PATH", path)
    config.clear_coaching_config_cache()
    assert config.get_coaching_config_model().coach.name == "Alexandre Mayer"

    raw["coach"]["name"] = "Reloaded Coach"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert config.get_coaching_config_model().coach.name == "Reloaded Coach"
    config.clear_coaching_config_cache()


def test_invalid_config_fails_validation():
    raw = yaml.safe_load(config.COACHING_CONFIG_PATH.read_text())
    raw.pop("zones")
    with pytest.raises(ValidationError):
        config.CoachingConfig.model_validate(raw)


def test_config_enforces_device_laps_without_auto_detected_fallback():
    policy = config.get_coaching_config()["interval_analysis"]
    assert policy == {
        "source": "device_laps",
        "allow_intervals_icu_detected": False,
        "on_missing_device_laps": "activity_level_only",
    }


def test_activity_detail_requests_positioned_intervals(monkeypatch):
    captured = {}

    def fake_get(path, params=None):
        captured.update(path=path, params=params)
        return {"id": "i123", "icu_intervals": []}

    monkeypatch.setattr(icu, "_get", fake_get)
    result = icu.get_activity_detail("athlete", "i123")
    assert result["id"] == "i123"
    assert captured == {
        "path": "/activity/i123",
        "params": {"intervals": "true"},
    }


def test_configured_ftp_selects_matching_indoor_and_outdoor_values():
    profile = {
        "sportSettings": [{
            "types": ["Ride", "VirtualRide"],
            "ftp": 290,
            "indoor_ftp": 275,
        }]
    }
    assert icu.configured_ftp_for_activity(profile, "Ride", False) == 290
    assert icu.configured_ftp_for_activity(profile, "VirtualRide", True) == 275
    assert icu.configured_ftp_for_activity(profile, "Run", False) is None


def _mock_summary_apis(monkeypatch, detail, profile=None):
    monkeypatch.setattr(icu, "get_activity_detail", lambda *_: detail)
    monkeypatch.setattr(
        icu,
        "get_activity_streams",
        lambda *_: [
            {"type": "watts", "data": [200.0] * 120},
            {"type": "heartrate", "data": [130.0] * 120},
            {"type": "cadence", "data": [85.0] * 120},
        ],
    )
    monkeypatch.setattr(icu, "get_planned_workout", lambda *_: None)
    monkeypatch.setattr(icu, "get_wellness", lambda *_: None)
    monkeypatch.setattr(icu, "get_athlete_profile", lambda *_: profile)


def _detail(**updates):
    result = {
        "start_date_local": "2026-07-14T08:00:00",
        "name": "Ride",
        "type": "Ride",
        "moving_time": 120,
        "elapsed_time": 120,
        "icu_average_watts": 200,
        "icu_weighted_avg_watts": 202,
        "icu_intensity": 75,
        "icu_training_load": 20,
    }
    result.update(updates)
    return result


def test_activity_ftp_wins_over_modeled_and_local_ftp(monkeypatch):
    _mock_summary_apis(
        monkeypatch,
        _detail(icu_ftp=300, icu_pm_ftp_watts=410),
        {"sportSettings": [{"types": ["Ride"], "ftp": 280}]},
    )
    summary = build_session_summary("athlete", "activity", {"ftp_W": 500})
    assert summary["ftp_W"] == 300
    assert summary["ftp_source"] == "activity"


def test_sport_settings_are_the_only_ftp_fallback(monkeypatch):
    _mock_summary_apis(
        monkeypatch,
        _detail(icu_pm_ftp_watts=410, trainer=True, type="VirtualRide"),
        {"sportSettings": [{"types": ["VirtualRide"], "ftp": 290, "indoor_ftp": 270}]},
    )
    summary = build_session_summary("athlete", "activity", {"ftp_W": 500})
    assert summary["ftp_W"] == 270
    assert summary["ftp_source"] == "sport_settings"


def test_missing_intervals_ftp_never_uses_modeled_local_or_250(monkeypatch):
    _mock_summary_apis(monkeypatch, _detail(icu_pm_ftp_watts=410), {"sportSettings": []})
    summary = build_session_summary("athlete", "activity", {"ftp_W": 500})
    assert summary["ftp_W"] is None
    assert summary["ftp_source"] == "missing"
    assert summary["if_value"] is None
    assert summary["tss"] is None
    assert summary["power_zone_times_s"] == {}


def test_classifier_prefers_planned_workout_and_reaches_summary():
    summary = {
        "name": "Easy spin",
        "planned_workout": {"name": "VO2 30 30", "description": ""},
        "if_value": 0.5,
        "avg_cadence_rpm": 90,
        "power_zone_times_s": {},
        "summary_version": 2,
        "activity_id": "a",
        "athlete_id": "u",
        "date": "2026-07-14",
        "type": "Ride",
        "is_indoor": False,
        "ftp_W": 300,
        "ftp_source": "activity",
        "hr_zone_times_s": {},
        "interval_details": [],
    }
    session_type = _classify_session(summary)
    enrich_session_summary(summary, session_type)
    assert summary["session_type"] == "vo2max"


@pytest.mark.parametrize("name", [
    "Girona - Torq sprint",
    "Torq-sprint efforts",
    "Torq+sprint session",
])
def test_torque_sprint_titles_match_torque_before_generic_sprint(name):
    assert _classify_session({
        "name": name,
        "planned_workout": {},
        "if_value": 1.1,
        "avg_cadence_rpm": 80,
    }) == "torque"


def test_zone_boundaries_are_continuous_and_hr_zones_are_separate():
    zones = config.get_coaching_config()["zones"]["coggan_7"]
    power = compute.time_in_zones_from_stream([55.0, 55.1, 75.0, 75.1], 100, zones)
    assert power["Active Recovery"] == 1
    assert power["Endurance"] == 2
    assert power["Tempo"] == 1
    hr = compute.time_in_hr_zones_from_stream([68.0, 68.1, 83.0, 83.1], 100, zones)
    assert hr["Active Recovery"] == 1
    assert hr["Endurance"] == 2
    assert hr["Tempo"] == 1


def test_recovery_short_power_spikes_with_low_hr_is_aerobically_easy():
    summary = {
        "summary_version": 2,
        "activity_id": "a",
        "athlete_id": "u",
        "date": "2026-07-14",
        "name": "Recovery",
        "type": "Ride",
        "is_indoor": False,
        "ftp_W": 300,
        "ftp_source": "activity",
        "power_zone_times_s": {"Active Recovery": 800, "Endurance": 100, "Tempo": 100},
        "hr_zone_times_s": {"Active Recovery": 900, "Endurance": 100},
        "longest_hr_above_z2_s": 20,
        "peak_30s_W": 250,
        "interval_details": [],
    }
    enrich_session_summary(summary, "recovery")
    assert summary["recovery_aerobically_easy"] is True


def test_interval_summary_is_not_used_as_device_lap_fallback():
    detail = {"interval_summary": ["3x 2m 300w"]}
    streams = {
        "watts": [100.0] * 1000,
        "heartrate": [120.0] * 1000,
        "cadence": [90.0] * 1000,
    }
    intervals = _extract_interval_list(detail, streams, 300, 300)
    assert intervals == []


def test_unverified_icu_auto_detected_intervals_are_rejected():
    detail = {
        "use_laps_for_power_intervals": False,
        "icu_intervals": [{
            "type": "WORK",
            "start_index": 0,
            "end_index": 59,
            "moving_time": 60,
            "average_watts": 300,
        }],
    }
    intervals = _extract_interval_list(detail, {"watts": [300.0] * 60}, 300, 300)
    assert intervals == []


def test_verified_lap_mode_uses_positioned_icu_intervals_and_stream_metrics():
    detail = {
        "use_laps_for_power_intervals": True,
        "icu_intervals": [
            {
                "type": "WORK",
                "start_index": 0,
                "end_index": 59,
                "moving_time": 60,
                "average_watts": 300,
                "weighted_average_watts": 301,
                "average_heartrate": 165,
                "average_cadence": 90,
                "average_torque": 32,
            },
            {
                "type": "RECOVERY",
                "start_index": 60,
                "end_index": 89,
                "moving_time": 30,
                "average_watts": 120,
            },
        ],
    }
    streams = {
        "watts": [300.0] * 60 + [120.0] * 30,
        "heartrate": [165.0] * 60 + [130.0] * 30,
        "cadence": [90.0] * 60 + [80.0] * 30,
        "torque": [32.0] * 60 + [15.0] * 30,
    }
    intervals = _extract_interval_list(detail, streams, 300, 300)
    assert len(intervals) == 2
    assert intervals[0]["source"] == "device_lap"
    assert intervals[0]["source_detail"] == "icu_intervals_lap_mode"
    assert intervals[0]["is_work"] is True
    assert intervals[0]["peak_30s_W"] == 300
    assert intervals[0]["avg_torque_Nm"] == 32
    assert intervals[1]["is_work"] is False


def test_missing_device_laps_preserves_activity_analysis_with_provenance(monkeypatch):
    _mock_summary_apis(
        monkeypatch,
        _detail(icu_ftp=300, interval_summary=["3x 2m 300w"]),
        {"sportSettings": [{"types": ["Ride"], "ftp": 300}]},
    )
    summary = build_session_summary("athlete", "activity", {})
    assert summary["interval_source"] == "missing"
    assert summary["interval_source_verified"] is False
    assert summary["interval_metrics_available"] is False
    assert summary["automatic_interval_detection_used"] is False
    assert summary["incomplete_data_reason"] == "device_laps_not_available"
    assert summary["interval_details"] == []
    assert summary["avg_power_W"] == 200


def test_prompt_consumes_configured_session_metadata_and_dynamic_structure():
    prompt = build_system_prompt()
    assert "Feedback structure (3 steps)" in prompt
    assert "Target zone: Z4" in prompt
    assert "Pacing preference: even_to_slight_negative_split" in prompt
    assert "torque_cadence_deviation_rpm" in prompt
    assert "Accepted interval source: device_laps only" in prompt
    assert "Missing device laps are incomplete data, not a performance failure" in prompt


def test_historical_settings_control_enablement_and_fields(monkeypatch):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(Athlete(id="u", name="Athlete"))
        db.add(Activity(
            id="old",
            athlete_id="u",
            date="2026-07-13",
            session_type="tempo",
            avg_power_W=210,
            session_summary_json={"time_in_target_zone_pct": 82.5},
        ))
        db.add(Analysis(
            activity_id="old",
            athlete_id="u",
            verdict="well",
        ))
        db.commit()

        history = {
            "enabled": True,
            "num_sessions": 3,
            "match_on": ["session_type"],
            "fields_to_compare": ["avg_power_W", "time_in_target_zone_pct", "verdict"],
        }
        monkeypatch.setattr(crud, "get_coaching_config", lambda: {"historical_comparison": history})
        result = crud.get_similar_sessions(db, "u", "tempo", "current")
        assert result == [{
            "date": "2026-07-13",
            "avg_power_W": 210,
            "time_in_target_zone_pct": 82.5,
            "verdict": "well",
        }]

        history["enabled"] = False
        assert crud.get_similar_sessions(db, "u", "tempo", "current") == []


def test_athlete_message_validation_enforces_style():
    _validate_athlete_message("Nice one today. Power was controlled throughout. Keep it up.")
    with pytest.raises(ValueError, match="forbidden dash"):
        _validate_athlete_message("Nice one today. Power was controlled — well done. Keep it up.")
    with pytest.raises(ValueError, match="exactly 3 sentences"):
        _validate_athlete_message("Nice one today. Keep it up.")


def test_analysis_retries_once_after_output_validation_failure(monkeypatch):
    invalid = {
        "verdict": "well",
        "key_observations": ["Steady power", "Low variability"],
        "reasoning": "Controlled session",
        "escalate_to_coach": {"flag": False, "reason": ""},
        "athlete_message": "Nice one today. Keep it up.",
    }
    valid = {
        **invalid,
        "athlete_message": "Nice one today. Power stayed controlled throughout. Keep it up.",
    }

    class FakeClient:
        def __init__(self):
            self.calls = 0
            self.messages = self

        def create(self, **kwargs):
            payload = invalid if self.calls == 0 else valid
            self.calls += 1
            return SimpleNamespace(
                content=[SimpleNamespace(text=json.dumps(payload))],
                usage=SimpleNamespace(
                    input_tokens=100,
                    output_tokens=50,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                ),
            )

    fake_client = FakeClient()
    monkeypatch.setattr(
        analyser,
        "get_settings",
        lambda: SimpleNamespace(anthropic_api_key="test"),
    )
    monkeypatch.setattr(analyser.anthropic, "Anthropic", lambda **_: fake_client)
    result = analyser.analyse_session(
        {
            "activity_id": "a",
            "athlete_id": "u",
            "ftp_W": 300,
            "ftp_source": "activity",
            "wellness": {},
        },
        {"name": "Athlete"},
        [],
    )
    assert fake_client.calls == 2
    assert result["athlete_message"] == valid["athlete_message"]
    assert result["_meta"]["input_tokens"] == 200
