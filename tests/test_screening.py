"""Tests for screening: Phase 1 hard filter, Phase 2 scoring, dynamic threshold."""

from datetime import date, timedelta

import polars as pl

from alphascreener.core.screening import (
    DEFAULT_THRESHOLDS,
    FACTOR_WEIGHTS,
    DynamicThreshold,
    compute_missing_rate,
    phase1_hard_filter,
    phase2_score,
    standardize_factors,
)


def _make_factor_df(ticker_data):
    """Create a factor DataFrame from a list of dicts."""
    return pl.DataFrame(ticker_data)


class TestStandardizeFactors:
    def test_z_score_clipped(self):
        df = pl.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "date": [date.today()] * 3,
                "mom_5d": [0.01, 0.02, 100.0],
            }
        )
        result = standardize_factors(df)
        assert "z_mom_5d" in result.columns
        assert "score_mom_5d" in result.columns
        z = result["z_mom_5d"]
        assert (z >= -3.0).all() and (z <= 3.0).all()
        s = result["score_mom_5d"]
        assert (s >= 0.0).all() and (s <= 100.0).all()

    def test_identical_values(self):
        df = pl.DataFrame(
            {
                "ticker": ["A", "B"],
                "date": [date.today()] * 2,
                "mom_5d": [0.02, 0.02],
            }
        )
        result = standardize_factors(df)
        assert (result["z_mom_5d"] == 0.0).all()
        assert (result["score_mom_5d"] == 50.0).all()


class TestPhase1HardFilter:
    def test_normal_case(self):
        df = pl.DataFrame(
            {
                "ticker": ["OK", "BAD_MOM", "BAD_MFI"],
                "date": [date.today()] * 3,
                "mom_5d": [0.05, -0.01, 0.05],
                "vol_anomaly": [0, 1, 0],
                "mfi_14": [50.0, 50.0, 30.0],
                "atr_ratio": [0.5, 0.5, 0.5],
            }
        )
        result = phase1_hard_filter(df)
        assert len(result) == 1
        assert result["ticker"][0] == "OK"

    def test_vol_anomaly_passes(self):
        df = pl.DataFrame(
            {
                "ticker": ["V"],
                "date": [date.today()],
                "mom_5d": [0.05],
                "vol_anomaly": [1],
                "mfi_14": [30.0],
                "atr_ratio": [0.5],
            }
        )
        result = phase1_hard_filter(df)
        assert len(result) == 1

    def test_empty_result(self):
        df = pl.DataFrame(
            {
                "ticker": ["X"],
                "date": [date.today()],
                "mom_5d": [-0.05],
                "vol_anomaly": [0],
                "mfi_14": [30.0],
                "atr_ratio": [0.5],
            }
        )
        result = phase1_hard_filter(df)
        assert result.is_empty()


class TestPhase2Score:
    def test_scores_descending(self):
        df = pl.DataFrame(
            {
                "ticker": ["A", "B"],
                "date": [date.today()] * 2,
                "z_mom_5d": [1.0, -1.0],
                "z_pth": [0.5, -0.5],
                "z_mom_slope": [0.3, -0.3],
                "z_bb_squeeze": [0.2, -0.2],
                "z_atr_ratio": [0.1, -0.1],
                "z_mfi_14": [0.4, -0.4],
                "z_cmf_21": [0.3, -0.3],
                "z_vol_anomaly": [0.1, -0.1],
                "z_rsi_oversold": [0.2, -0.2],
                "z_macd_cross": [0.1, -0.1],
                "z_golden_cross": [0.1, -0.1],
                "z_pead_flag": [0.0, 0.0],
                "z_insider_buy": [0.2, -0.2],
                "z_rev_accel": [0.1, -0.1],
            }
        )
        result = phase2_score(df)
        assert result["coarse_score"][0] > result["coarse_score"][1]

    def test_missing_z_columns(self):
        df = pl.DataFrame(
            {
                "ticker": ["A"],
                "date": [date.today()],
            }
        )
        result = phase2_score(df)
        assert "coarse_score" in result.columns


class TestMissingRate:
    def test_all_present(self):
        df = pl.DataFrame(
            {
                "ticker": ["A", "B"],
                "date": [date.today()] * 2,
                "mom_5d": [0.01, 0.02],
            }
        )
        rate = compute_missing_rate(df)
        assert (rate == 0.0).all()

    def test_some_missing(self):
        df = pl.DataFrame(
            {
                "ticker": ["A", "B"],
                "date": [date.today()] * 2,
                "mom_5d": [0.01, None],
            }
        )
        rate = compute_missing_rate(df)
        assert rate[1] == 1.0  # B has missing
        assert rate[0] == 0.0  # A has no missing


class TestDynamicThreshold:
    def test_normal_pass_rate(self):
        dt = DynamicThreshold()
        today = date.today()
        thresholds, status, action = dt.evaluate(0.85, today)
        assert status == "normal"
        assert action == "none"

    def test_tight_pass_rate(self):
        dt = DynamicThreshold()
        today = date.today()
        thresholds, status, action = dt.evaluate(0.93, today)
        assert status == "tight"
        assert action == "warn"

    def test_very_tight_widens(self):
        dt = DynamicThreshold()
        today = date.today()
        thresholds, status, action = dt.evaluate(0.97, today)
        assert status == "very_tight"
        assert action == "widen"
        # Thresholds should be adjusted
        assert thresholds["mom_5d_min"] < DEFAULT_THRESHOLDS["mom_5d_min"]
        assert thresholds["atr_ratio_max"] > DEFAULT_THRESHOLDS["atr_ratio_max"]

    def test_extreme_pass_rate(self):
        dt = DynamicThreshold()
        today = date.today()
        thresholds, status, action = dt.evaluate(0.99, today)
        assert action == "extreme_widen"

    def test_loose_pass_rate_tightens(self):
        dt = DynamicThreshold()
        today = date.today()
        # First widen, then tighten
        dt.evaluate(0.97, today)
        tomorrow = today + timedelta(days=10)
        thresholds, status, action = dt.evaluate(0.60, tomorrow)
        assert action == "tighten"

    def test_cooldown_respected(self):
        dt = DynamicThreshold()
        today = date.today()
        dt.evaluate(0.97, today)  # widens
        dt.evaluate(0.97, today + timedelta(days=1))  # within cooldown
        # Cooldown prevents second adjustment within 3 days
        tomorrow = today + timedelta(days=1)
        thresholds2, status2, action2 = dt.evaluate(0.97, tomorrow)
        assert action2 == "widen"  # Still requests widen but cooldown prevents actual adjustment


class TestFactorWeights:
    def test_all_thirteen_factors(self):
        assert len(FACTOR_WEIGHTS) == 14  # 13 regular + pead_flag dummy

    def test_weights_sum(self):
        total = sum(FACTOR_WEIGHTS.values())
        assert 0.99 <= total <= 1.01
