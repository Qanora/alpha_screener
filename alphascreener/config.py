"""Application configuration via pydantic-settings."""

import functools
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    primary_data_source: str = "yfinance"
    fallback_ohlcv_source: str = "stooq"
    fmp_tier: str = "free"
    fmp_daily_budget: int = 250

    llm_model: str = "gpt-4o-mini"
    llm_rps: int = 5
    llm_batch_size: int = 3
    llm_max_concurrent_stage1: int = 6

    yfinance_rps: int = 5
    yfinance_batch_size: int = 50

    cost_l1_warning_daily_usd: float = 0.80
    cost_l2_degrade_daily_usd: float = 1.00
    cost_l3_savings_monthly_avg_usd: float = 2.67
    cost_l4_circuit_monthly_avg_usd: float = 3.17

    mom_5d_min: float = 0.0
    atr_ratio_max: float = 0.8
    rsi_lower: float = 25.0
    rsi_upper: float = 75.0
    mfi_min_or_vol_anomaly: float = 40.0

    sector_cap: int = 3
    industry_cap: int = 2

    evolution_weight_adjust_enabled: bool = False
    llm_ablation_enabled: bool = True
    cost_budget_monthly_usd: int = 100

    alphascreener_home: str = "~/.alphascreener"

    @field_validator("alphascreener_home", mode="before")
    @classmethod
    def expand_home(cls, v: str) -> str:
        return str(Path(v).expanduser())

    @property
    def home(self) -> Path:
        return Path(self.alphascreener_home)

    @property
    def db_path(self) -> Path:
        return self.home / "db" / "metadata.db"


@functools.lru_cache
def get_settings() -> Settings:
    return Settings()
