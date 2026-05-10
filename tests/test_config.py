"""Tests for application configuration."""

from pathlib import Path

from alphascreener.config import Settings, get_settings


class TestSettingsDefaults:
    def test_data_source_defaults(self):
        s = Settings()
        assert s.primary_data_source == "yfinance"
        assert s.fallback_ohlcv_source == "stooq"
        assert s.fmp_tier == "free"
        assert s.fmp_daily_budget == 250

    def test_llm_defaults(self):
        s = Settings()
        assert s.llm_model == "gpt-4o-mini"
        assert s.llm_rps == 5
        assert s.llm_batch_size == 3

    def test_cost_thresholds(self):
        s = Settings()
        assert s.cost_l1_warning_daily_usd == 0.80
        assert s.cost_l4_circuit_monthly_avg_usd == 3.17

    def test_screening_thresholds(self):
        s = Settings()
        assert s.mom_5d_min == 0.0
        assert s.atr_ratio_max == 0.8
        assert s.rsi_lower == 25.0
        assert s.rsi_upper == 75.0

    def test_behavior_switches(self):
        s = Settings()
        assert s.evolution_weight_adjust_enabled is False
        assert s.llm_ablation_enabled is True
        assert s.cost_budget_monthly_usd == 100

    def test_home_expansion(self):
        s = Settings(alphascreener_home="~/.test_alpha")
        assert str(s.alphascreener_home).endswith(".test_alpha")
        assert s.home == Path(s.alphascreener_home)

    def test_db_path(self):
        s = Settings(alphascreener_home="/tmp/test_home")
        assert s.db_path == Path("/tmp/test_home/db/metadata.db")

    def test_yfinance_settings(self):
        s = Settings()
        assert s.yfinance_rps == 5
        assert s.yfinance_batch_size == 50

    def test_env_override(self):
        s = Settings(mom_5d_min=0.02, atr_ratio_max=0.75)
        assert s.mom_5d_min == 0.02
        assert s.atr_ratio_max == 0.75


class TestGetSettings:
    def test_returns_settings(self):
        s = get_settings()
        assert isinstance(s, Settings)

    def test_singleton(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
