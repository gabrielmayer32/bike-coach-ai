from __future__ import annotations
"""SQLAlchemy ORM models."""

import json
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, Text, JSON,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Athlete(Base):
    __tablename__ = "athletes"

    id = Column(String, primary_key=True)          # Intervals.icu athlete_id e.g. "i93850"
    name = Column(String, nullable=False)
    email = Column(String)

    # Physiological profile
    ftp_W = Column(Float)                           # Current FTP in watts
    lthr_bpm = Column(Float)                        # Lactate threshold HR
    max_hr_bpm = Column(Float)
    weight_kg = Column(Float)
    training_age_years = Column(Float)
    level = Column(String, default="intermediate")  # beginner / intermediate / advanced / elite
    training_phase = Column(String, default="build") # base / build / peak / taper

    # Config overrides (JSON blob) — allows per-athlete tolerance tuning later
    config_overrides = Column(JSON, default=dict)
    notes = Column(Text, default="")

    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    activities = relationship("Activity", back_populates="athlete", lazy="dynamic")


class Activity(Base):
    __tablename__ = "activities"

    id = Column(String, primary_key=True)           # Intervals.icu activity id
    athlete_id = Column(String, ForeignKey("athletes.id"), nullable=False)

    date = Column(String)                           # "YYYY-MM-DD"
    name = Column(String)
    activity_type = Column(String)                  # Ride, Run, etc.
    session_type = Column(String)                   # our classification: endurance_z2, threshold, etc.
    source = Column(String)                         # GARMIN_CONNECT, etc.
    is_indoor = Column(Boolean, default=False)

    # Computed metrics (stored so we don't re-fetch for historical comparison)
    avg_power_W = Column(Float)
    np_W = Column(Float)
    ftp_W = Column(Float)
    if_value = Column(Float)
    tss = Column(Float)
    vi = Column(Float)
    decoupling_pct = Column(Float)
    rep_fade_pct = Column(Float)
    avg_hr_bpm = Column(Float)
    avg_cadence_rpm = Column(Float)
    duration_s = Column(Integer)
    kj = Column(Float)
    rpe = Column(Float)
    temp_c = Column(Float)

    # Full computed summary (JSON) — for re-analysis without re-fetching
    session_summary_json = Column(JSON)

    fetched_at = Column(DateTime, default=datetime.utcnow)

    athlete = relationship("Athlete", back_populates="activities")
    analyses = relationship("Analysis", back_populates="activity", lazy="dynamic")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    activity_id = Column(String, ForeignKey("activities.id"), nullable=False)
    athlete_id = Column(String, nullable=False)

    # AI output
    verdict = Column(String)                        # well / okay / poor
    key_observations = Column(JSON)                 # list of strings
    reasoning = Column(Text)
    athlete_message = Column(Text)
    escalate_flag = Column(Boolean, default=False)
    escalate_reason = Column(Text)

    # Cost & metadata
    model = Column(String)
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    cache_write_tokens = Column(Integer)
    cache_read_tokens = Column(Integer)
    cost_usd = Column(Float)

    # Full AI input/output stored for debugging and reanalysis comparison
    input_summary_json = Column(JSON)
    raw_output_json = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)

    activity = relationship("Activity", back_populates="analyses")
