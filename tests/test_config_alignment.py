from __future__ import annotations

import os
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app import config
from app.ai import analyser
from app.ai.analyser import (
    AnalysisResult,
    _validate_analysis_grounding,
    _validate_athlete_message,
)
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
        "ignore_recovery_fragments_shorter_than_s": 10.0,
    }


def test_every_session_type_owns_a_recognition_signature():
    cfg = config.get_coaching_config()
    assert cfg["classification"] == {"default_session_type": "unclassified"}
    assert set(cfg["session_types"]) == {
        "unclassified",
        "recovery",
        "endurance_z2",
        "tempo",
        "threshold",
        "over_unders",
        "vo2max",
        "sprint_neuromuscular",
        "torque",
    }
    assert all(
        session["recognition"].get("mode")
        for session in cfg["session_types"].values()
    )
    policy = cfg["inferred_session_policy"]
    assert policy["missing_plan_must_not_reduce_verdict"] is True
    assert policy["allow_well_for_supported_observed_execution"] is True
    assert "well verdict is still allowed" in cfg["verdict"]["well"]["description"]
    assert "Missing or unverified data can never" in cfg["verdict"]["poor"]["description"]


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


def test_athlete_profile_uses_current_settings_endpoint(monkeypatch):
    captured = {}

    def fake_get(path, params=None):
        captured.update(path=path, params=params)
        return {"sportSettings": [{"types": ["Ride"], "ftp": 310}]}

    monkeypatch.setattr(icu, "_get", fake_get)
    profile = icu.get_athlete_profile("i123")
    assert profile["sportSettings"][0]["ftp"] == 310
    assert captured == {"path": "/athlete/i123", "params": None}


def test_current_ride_ftp_uses_profile_setting_not_eftp(monkeypatch):
    monkeypatch.setattr(
        icu,
        "get_athlete_profile",
        lambda *_: {
            "sportSettings": [{
                "types": ["Ride", "VirtualRide"],
                "ftp": 240,
                "indoor_ftp": 230,
                "eftp": 275,
            }]
        },
    )
    assert icu.get_current_ride_ftp("i123") == 240
    assert icu.get_current_ride_ftp("i123", is_indoor=True) == 230


def test_power_curve_data_does_not_relabel_modeled_ftp_as_profile_ftp(monkeypatch):
    monkeypatch.setattr(
        icu,
        "_get",
        lambda *_args, **_kwargs: {
            "list": [{
                "secs": [5, 60],
                "watts": [900, 400],
                "watts_per_kg": [12, 5.3],
                "powerModels": [{"ftp": 275}],
            }]
        },
    )
    curve = icu.get_power_curve_range("i123", since=date(2026, 1, 1))
    assert curve == {
        "secs": [5, 60],
        "watts": [900, 400],
        "w_per_kg": [12, 5.3],
    }


def test_athlete_template_displays_live_profile_ftp_not_cached_activity_ftp():
    template = Path("app/templates/athlete.html").read_text()
    assert "{{ profile_ftp_W|int }}W" in template
    assert "FTP · Intervals profile" in template
    assert "{{ athlete.ftp_W|int }}W" not in template


def test_reanalysis_polling_preserves_analysis_id_across_reload_and_retry():
    template = Path("app/templates/athlete.html").read_text()
    assert "afterId: currentAnalysisId" in template
    assert "for (const {id: actId, afterId} of inflight)" in template
    assert "pollForResult(actId, afterId)" in template
    assert "const retryCall = afterId != null" in template
    assert '? "triggerReanalyse(\'" + ATHLETE_ID' in template


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
        "icu_lap_count": 4,
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


def test_reported_lap_count_recognizes_lap_intervals_without_legacy_flag():
    detail = {
        "icu_lap_count": 4,
        "icu_intervals": [
            {
                "type": "WORK",
                "start_index": 0,
                "end_index": 119,
                "moving_time": 120,
                "average_watts": 280,
                "average_cadence": 60,
            },
            {
                "type": "RECOVERY",
                "start_index": 120,
                "end_index": 179,
                "moving_time": 60,
                "average_watts": 120,
                "average_cadence": 80,
            },
        ],
    }
    streams = {
        "watts": [280.0] * 120 + [120.0] * 60,
        "cadence": [60.0] * 120 + [80.0] * 60,
    }
    intervals = _extract_interval_list(detail, streams, 300, None)
    assert len(intervals) == 2
    assert intervals[0]["source"] == "device_lap"
    assert intervals[0]["source_detail"] == "icu_intervals_lap_count"


def test_more_detected_intervals_than_reported_laps_are_rejected():
    interval = {
        "type": "WORK",
        "start_index": 0,
        "end_index": 59,
        "moving_time": 60,
        "average_watts": 300,
    }
    detail = {"icu_lap_count": 2, "icu_intervals": [interval] * 3}
    assert _extract_interval_list(detail, {"watts": [300.0] * 60}, 300, None) == []


def test_tiny_residual_fragment_does_not_hide_five_device_laps():
    meaningful = [
        {
            "type": "WORK",
            "start_index": index * 100,
            "end_index": (index + 1) * 100,
            "moving_time": duration,
            "average_watts": power,
        }
        for index, (duration, power) in enumerate([
            (1231, 196),
            (1800, 308),
            (600, 220),
            (1801, 296),
            (4384, 180),
        ])
    ]
    detail = {
        "icu_lap_count": 5,
        "icu_intervals": meaningful + [
            {"type": "RECOVERY", "moving_time": 6, "average_watts": None},
            {"type": "RECOVERY", "moving_time": 3, "average_watts": None},
        ],
    }
    intervals = _extract_interval_list(detail, {}, 350, None)
    assert len(intervals) == 5
    assert all(interval["duration_s"] >= 10 for interval in intervals)
    assert all(interval["is_target_work"] is None for interval in intervals)


def test_short_work_sprint_is_not_removed_as_a_residual_fragment():
    detail = {
        "icu_lap_count": 2,
        "icu_intervals": [
            {"type": "WORK", "moving_time": 3, "average_watts": 900},
            {"type": "RECOVERY", "moving_time": 30, "average_watts": 100},
        ],
    }
    intervals = _extract_interval_list(detail, {}, 300, None)
    assert [interval["duration_s"] for interval in intervals] == [3, 30]


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


def test_low_cadence_device_work_laps_classify_torque_over_if_fallback():
    summary = {
        "name": "Aigle Road Cycling",
        "planned_workout": {},
        "ftp_W": 300,
        "if_value": 0.755,
        "avg_cadence_rpm": 86,
        "interval_source": "device_laps",
        "interval_details": [
            {
                "is_work": True,
                "duration_s": 249,
                "avg_power_W": 272,
                "avg_cadence_rpm": 64.9,
            },
            {
                "is_work": True,
                "duration_s": 131,
                "avg_power_W": 262,
                "avg_cadence_rpm": 55.3,
            },
        ],
    }
    assert _classify_session(summary) == "torque"


def test_two_comparable_30m_tempo_intervals_override_whole_ride_if():
    summary = {
        "name": "CHAUD",
        "planned_workout": {},
        "ftp_W": 350,
        "if_value": 0.723,
        "avg_cadence_rpm": 87.8,
        "interval_source": "device_laps_inferred",
        "interval_details": [
            {"duration_s": 1408, "avg_power_W": 196, "is_work": True},
            {"duration_s": 1919, "avg_power_W": 308, "is_work": True},
            {"duration_s": 793, "avg_power_W": 220, "is_work": True},
            {"duration_s": 1801, "avg_power_W": 296, "is_work": True},
            {"duration_s": 5067, "avg_power_W": 180, "is_work": True},
        ],
    }
    assert _classify_session(summary) == "tempo"
    assert summary["session_type_source"] == "configured_interval_signature"
    assert summary["session_type_evidence"] == {
        "signature_session_type": "tempo",
        "matched_interval_count": 2,
        "interval_indices": [1, 3],
        "durations_s": [1919, 1801],
        "avg_power_W": [308, 296],
        "power_pct_ftp": [88.0, 84.6],
        "avg_cadence_rpm": [None, None],
        "duration_spread_pct": 6.3,
        "power_spread_pct": 4.0,
    }
    assert summary["interval_details"][1]["observed_role"] == "tempo_signature_match"
    assert summary["interval_details"][3]["observed_role"] == "tempo_signature_match"
    assert summary["interval_details"][0]["observed_role"] == "unknown"
    assert summary["analysis_constraints"]["whole_activity_metrics_verdict_eligible"] is False


def test_non_comparable_long_efforts_do_not_infer_tempo_pattern():
    summary = {
        "name": "Road ride",
        "planned_workout": {},
        "ftp_W": 350,
        "if_value": 0.723,
        "avg_cadence_rpm": 88,
        "interval_source": "device_laps_inferred",
        "interval_details": [
            {"duration_s": 1800, "avg_power_W": 308, "is_work": True},
            {"duration_s": 1200, "avg_power_W": 282, "is_work": True},
        ],
    }
    assert _classify_session(summary) == "endurance_z2"
    assert summary["session_type_source"] == "configured_whole_activity_if"


def test_over_unders_title_tolerates_spaced_hyphens_and_plural():
    summary = {
        "name": "Black River - Over - Unders",
        "planned_workout": {},
        "ftp_W": 232,
        "if_value": 0.849,
        "target_interval_membership_verified": False,
        "interval_details": [],
    }
    assert _classify_session(summary) == "over_unders"
    assert summary["session_type_source"] == "activity_text"
    assert summary["analysis_constraints"]["planned_roles_verified"] is False


@pytest.mark.parametrize(
    ("session_type", "duration_s", "powers"),
    [
        ("threshold", 900, [290, 295]),
        ("vo2max", 240, [330, 335, 340]),
        ("sprint_neuromuscular", 10, [500, 520]),
    ],
)
def test_generic_interval_signatures_classify_supported_sessions(
    session_type, duration_s, powers
):
    summary = {
        "name": "Road ride",
        "planned_workout": {},
        "ftp_W": 300,
        "if_value": 0.7,
        "interval_source": "device_laps_inferred",
        "interval_details": [
            {
                "duration_s": duration_s,
                "avg_power_W": power,
                "avg_cadence_rpm": 90,
                "is_work": True,
            }
            for power in powers
        ],
    }
    assert _classify_session(summary) == session_type
    assert summary["session_type_source"] == "configured_interval_signature"
    assert summary["session_type_evidence"]["signature_session_type"] == session_type


@pytest.mark.parametrize(
    ("if_value", "session_type"),
    [
        (0.50, "recovery"),
        (0.70, "endurance_z2"),
        (0.80, "tempo"),
        (0.95, "threshold"),
        (1.10, "vo2max"),
        (1.30, "sprint_neuromuscular"),
    ],
)
def test_configured_whole_activity_if_ranges_are_generic_fallbacks(
    if_value, session_type
):
    summary = {
        "name": "Road ride",
        "planned_workout": {},
        "ftp_W": 300,
        "if_value": if_value,
        "interval_details": [],
    }
    assert _classify_session(summary) == session_type
    assert summary["session_type_source"] == "configured_whole_activity_if"


def test_missing_classification_evidence_returns_unclassified():
    summary = {
        "name": "Road ride",
        "planned_workout": {},
        "ftp_W": None,
        "if_value": None,
        "interval_details": [],
    }
    assert _classify_session(summary) == "unclassified"
    assert summary["session_type_source"] == "unclassified_fallback"


def test_text_only_signature_can_name_over_unders_without_phase_claims():
    summary = {
        "name": "Over-under road session",
        "planned_workout": {},
        "ftp_W": 300,
        "if_value": 0.7,
        "interval_details": [],
    }
    assert _classify_session(summary) == "over_unders"
    assert summary["session_type_source"] == "activity_text"


def test_keyword_matching_does_not_match_substrings():
    summary = {
        "name": "Visit the coast",
        "planned_workout": {},
        "ftp_W": 300,
        "if_value": 0.7,
        "interval_details": [],
    }
    assert _classify_session(summary) == "endurance_z2"
    assert summary["session_type_source"] == "configured_whole_activity_if"


def test_planned_description_recovery_phase_does_not_override_workout_name():
    summary = {
        "name": "Road ride",
        "planned_workout": {
            "name": "VO2 5x4",
            "description": "Warm up easy, then recover between hard efforts",
        },
        "ftp_W": 300,
        "if_value": 0.7,
        "interval_details": [],
    }
    assert _classify_session(summary) == "vo2max"
    assert summary["session_type_source"] == "planned_workout_name"


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


def test_summary_exposes_reported_device_lap_count(monkeypatch):
    _mock_summary_apis(
        monkeypatch,
        _detail(
            icu_ftp=300,
            icu_lap_count=4,
            icu_intervals=[{
                "type": "WORK",
                "start_index": 0,
                "end_index": 119,
                "moving_time": 120,
                "average_watts": 280,
                "average_cadence": 60,
            }],
        ),
        {"sportSettings": [{"types": ["Ride"], "ftp": 300}]},
    )
    summary = build_session_summary("athlete", "activity", {})
    assert summary["summary_version"] == 5
    assert summary["device_lap_count"] == 4
    assert summary["interval_source"] == "device_laps_inferred"
    assert summary["interval_source_detail"] == "icu_intervals_lap_count"
    assert summary["interval_source_verified"] is False
    assert summary["automatic_interval_detection_used"] is None
    assert summary["target_interval_membership_verified"] is False
    assert summary["rep_metrics_available"] is False
    assert summary["rep_fade_pct"] is None
    assert summary["incomplete_data_reason"] == "planned_interval_membership_not_verified"


def test_unlinked_work_labels_do_not_create_torque_penalties():
    summary = {
        "summary_version": 5,
        "activity_id": "activity",
        "athlete_id": "athlete",
        "date": "2026-07-15",
        "name": "Road ride",
        "type": "Ride",
        "is_indoor": False,
        "ftp_W": 300,
        "ftp_source": "activity",
        "interval_details": [
            {"is_work": True, "is_target_work": None, "avg_cadence_rpm": 55},
            {"is_work": True, "is_target_work": None, "avg_cadence_rpm": 85},
        ],
        "power_zone_times_s": {},
        "hr_zone_times_s": {},
        "avg_cadence_rpm": 86,
    }
    enrich_session_summary(summary, "torque")
    assert summary["torque_cadence_deviation_rpm"] is None


def test_one_to_one_planned_steps_enable_rep_metrics(monkeypatch):
    detail = _detail(
        icu_ftp=300,
        use_laps_for_power_intervals=True,
        icu_intervals=[
            {
                "type": "WORK",
                "start_index": 0,
                "end_index": 59,
                "moving_time": 60,
                "average_watts": 300,
            },
            {
                "type": "WORK",
                "start_index": 60,
                "end_index": 119,
                "moving_time": 60,
                "average_watts": 285,
            },
        ],
    )
    _mock_summary_apis(
        monkeypatch,
        detail,
        {"sportSettings": [{"types": ["Ride"], "ftp": 300}]},
    )
    monkeypatch.setattr(
        icu,
        "get_planned_workout",
        lambda *_: {
            "name": "Threshold reps",
            "workout_doc": {
                "steps": [
                    {"target_watts": 300},
                    {"target_watts": 290},
                ]
            },
        },
    )
    summary = build_session_summary("athlete", "activity", {})
    assert summary["target_interval_membership_verified"] is True
    assert summary["rep_metrics_available"] is True
    assert summary["target_interval_count"] == 2
    assert summary["rep_fade_pct"] == 5.0
    assert [
        interval["target_membership_source"]
        for interval in summary["interval_details"]
    ] == ["planned_step_count", "planned_step_count"]


def test_prompt_consumes_configured_session_metadata_and_dynamic_structure():
    prompt = build_system_prompt()
    assert "Feedback structure (3 steps)" in prompt
    assert "Target zone: Z4" in prompt
    assert "Pacing preference: even_to_slight_negative_split" in prompt
    assert "torque_cadence_deviation_rpm" in prompt
    assert "Accepted interval source: device_laps only" in prompt
    assert "Missing device laps are incomplete data, not a performance failure" in prompt
    assert "generic interval is_work" in prompt
    assert "target_interval_membership_verified is false" in prompt
    assert "session_type_source=configured_interval_signature" in prompt
    assert "Recognition signature:" in prompt
    assert "missing or unlinked plan is missing context" in prompt
    assert "Two observed intervals match the configured tempo signature" in prompt


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


def _analysis_result(reasoning: str, observations: list[str]) -> AnalysisResult:
    return AnalysisResult.model_validate({
        "verdict": "okay",
        "key_observations": observations,
        "reasoning": reasoning,
        "escalate_to_coach": {"flag": False, "reason": ""},
        "athlete_message": "Nice one today. The observed tempo efforts were steady. Keep building.",
    })


def test_grounding_rejects_invented_roles_and_context_metric_penalties():
    result = _analysis_result(
        (
            "The warm-up lap, recovery lap and cool-down final lap created mixed structure. "
            "The work efforts were clean. Verdict is okay rather than well because target "
            "membership is unverified and whole-activity VI is high."
        ),
        [
            "Two tempo intervals were observed.",
            "Whole-activity decoupling is elevated.",
        ],
    )
    with pytest.raises(ValueError) as exc:
        _validate_analysis_grounding(result, {
            "session_type_source": "configured_interval_signature",
            "target_interval_membership_verified": False,
        })
    message = str(exc.value)
    assert "unverified interval phase label" in message
    assert "unqualified planned-role term" in message
    assert "must not reduce the verdict" in message


def test_grounding_accepts_required_inferred_session_wording():
    result = _analysis_result(
        (
            "Two observed intervals match the configured tempo signature and show steady "
            "execution. Because the planned workout is not linked, they cannot be confirmed "
            "as prescribed work intervals. The remaining lap roles are unknown, and "
            "whole-activity VI and decoupling are context only across this mixed-intensity "
            "structure."
        ),
        [
            "The two observed tempo signature matches were steady.",
            "The remaining interval roles are unknown.",
        ],
    )
    _validate_analysis_grounding(result, {
        "session_type_source": "configured_interval_signature",
        "target_interval_membership_verified": False,
    })


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
