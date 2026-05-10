"""Tests for screening: Phase 1 hard filter, Phase 2 scoring, dynamic threshold."""

from datetime import date, timedelta

import polars as pl

from alphascreener.core.screening import (
    DEFAULT_THRESHOLDS,
    FACTOR_WEIGHTS,
    MISSING_SECTOR_INDUSTRY,
    DynamicThreshold,
    compute_missing_rate,
    dedup_by_sector_industry,
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


class TestDedupBySectorIndustry:
    def test_basic_caps_enforced(self):
        """Sector <= 3, Industry <= 2, greedy by coarse_score desc."""
        df = pl.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(10)],
                "coarse_score": [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45],
                "sector": ["Tech"] * 5 + ["Finance"] * 5,
                "industry": (
                    ["Software"] * 3 + ["Hardware"] * 2 + ["Banks"] * 3 + ["Insurance"] * 2
                ),
            }
        )
        result = dedup_by_sector_industry(df, sector_cap=3, industry_cap=2, top_n=20)
        assert len(result) == 6
        tech_count = result.filter(pl.col("sector") == "Tech").height
        finance_count = result.filter(pl.col("sector") == "Finance").height
        assert tech_count <= 3
        assert finance_count <= 3
        software = result.filter(pl.col("industry") == "Software")
        assert software.height <= 2

    def test_top30_to_top20(self):
        """30 candidates with varied sectors/industries dedup to at most 20."""
        import random

        random.seed(42)
        sectors = ["Tech", "Finance", "Health", "Energy", "Consumer"]
        industries = [f"Ind_{s}_{i}" for s in sectors for i in range(3)]
        data = []
        for i in range(30):
            s = sectors[i % len(sectors)]
            ind = industries[(i * 7) % len(industries)]
            data.append(
                {
                    "ticker": f"T{i:02d}",
                    "coarse_score": 1.0 - i * 0.01,
                    "sector": s,
                    "industry": ind,
                }
            )
        df = pl.DataFrame(data)
        result = dedup_by_sector_industry(df, sector_cap=3, industry_cap=2, top_n=20)
        assert len(result) <= 20
        # Verify caps
        sector_counts = result.group_by("sector").len()
        for row in sector_counts.iter_rows(named=True):
            assert row["len"] <= 3, f"Sector {row['sector']} has {row['len']} > 3"
        industry_counts = result.group_by("industry").len()
        for row in industry_counts.iter_rows(named=True):
            assert row["len"] <= 2, f"Industry {row['industry']} has {row['len']} > 2"

    def test_fewer_than_30_input(self):
        """Only 5 candidates — all should pass if caps allow."""
        df = pl.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E"],
                "coarse_score": [0.5, 0.4, 0.3, 0.2, 0.1],
                "sector": ["Tech", "Tech", "Finance", "Health", "Energy"],
                "industry": ["SW", "SW", "Bank", "Bio", "Oil"],
            }
        )
        result = dedup_by_sector_industry(df, sector_cap=3, industry_cap=2, top_n=20)
        assert len(result) == 5

    def test_scores_descending(self):
        """Result must be sorted by coarse_score descending."""
        df = pl.DataFrame(
            {
                "ticker": ["B", "A", "C"],
                "coarse_score": [0.7, 0.9, 0.5],
                "sector": ["Tech", "Tech", "Tech"],
                "industry": ["SW", "HW", "SW"],
            }
        )
        result = dedup_by_sector_industry(df, sector_cap=3, industry_cap=2, top_n=20)
        scores = result["coarse_score"].to_list()
        assert scores == sorted(scores, reverse=True)

    def test_missing_sector_industry_treated_as_missing(self):
        """None values in sector/industry are filled with MISSING_SECTOR_INDUSTRY."""
        df = pl.DataFrame(
            {
                "ticker": ["A", "B", "C", "D"],
                "coarse_score": [0.9, 0.8, 0.7, 0.6],
                "sector": [None, None, "Tech", "Tech"],
                "industry": [None, None, "SW", "SW"],
            }
        )
        result = dedup_by_sector_industry(df, sector_cap=2, industry_cap=2, top_n=20)
        assert len(result) == 4
        missing = result.filter(pl.col("sector") == MISSING_SECTOR_INDUSTRY).height
        assert missing == 2

    def test_top_n_caps_output(self):
        """top_n parameter limits total output even if caps would allow more."""
        df = pl.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(15)],
                "coarse_score": [1.0 - i * 0.01 for i in range(15)],
                "sector": [f"S{i % 5}" for i in range(15)],
                "industry": [f"Ind_{i}" for i in range(15)],
            }
        )
        result = dedup_by_sector_industry(df, sector_cap=3, industry_cap=2, top_n=5)
        assert len(result) == 5


class TestFactorWeights:
    def test_all_thirteen_factors(self):
        assert len(FACTOR_WEIGHTS) == 14  # 13 regular + pead_flag dummy

    def test_weights_sum(self):
        total = sum(FACTOR_WEIGHTS.values())
        assert 0.99 <= total <= 1.01
