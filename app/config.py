from __future__ import annotations
"""App-wide settings and coaching config loader."""

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    intervals_api_key: str
    anthropic_api_key: str
    database_url: str = "sqlite:///./bike_coach.db"
    poll_interval_seconds: int = 300
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_coaching_config() -> dict:
    config_path = Path(__file__).parent.parent / "coaching_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
