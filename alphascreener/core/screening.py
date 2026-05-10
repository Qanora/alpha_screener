"""Phase 1 hard filtering, Phase 2 weighted scoring, dynamic threshold."""

import logging
from datetime import date
from typing import Tuple

import polars as pl

logger = logging.getLogger(__name__)

FACTOR_WEIGHTS = {
    "mom_5d": 0.14,
    "pth": 0.12,
    "mom_slope": 0.10,
    "bb_squeeze": 0.13,
    "atr_ratio": 0.09,
    "mfi_14": 0.095,
    "cmf_21": 0.085,
    "vol_anomaly": 0.045,
    "rsi_oversold": 0.045,
    "macd_cross": 0.035,
    "golden_cross": 0.035,
    "pead_flag": 0.0,
    "insider_buy": 0.045,
    "rev_accel": 0.035,
}

Z_SCORE_CLIP = 3.0
Z_SCORE_SCALE = 50.0 / 3.0
MISSING_RATE_MAX = 0.30
MISSING_SECTOR_INDUSTRY = "<MISSING>"
PASS_RATE_MIN = 0.70
PASS_RATE_NORMAL_LO = 0.80
PASS_RATE_NORMAL_HI = 0.92
PASS_RATE_TIGHT = 0.95
PASS_RATE_VERY_TIGHT = 0.98

DEFAULT_THRESHOLDS = {
    "mom_5d_min": 0.0,
    "mfi_14_min": 40.0,
    "atr_ratio_max": 0.8,
    "rsi_lower": 25.0,
    "rsi_upper": 75.0,
}


def standardize_factors(factor_df: pl.DataFrame) -> pl.DataFrame:
    """Z-score standardize all factor columns, clip to [-3, +3]."""
    factor_cols = [c for c in factor_df.columns if c not in ("ticker", "date")]
    result = factor_df.select(["ticker", "date"])

    for col_name in factor_cols:
        col = factor_df[col_name]
        mean = col.mean()
        std = col.std()
        if std is None or std == 0:
            result = result.with_columns(pl.lit(0.0).alias(f"z_{col_name}"))
            result = result.with_columns(pl.lit(50.0).alias(f"score_{col_name}"))
            continue

        z = (col - mean) / std
        z_capped = z.clip(-Z_SCORE_CLIP, Z_SCORE_CLIP)
        score = 50.0 + z_capped * Z_SCORE_SCALE
        result = result.with_columns(z_capped.alias(f"z_{col_name}"))
        result = result.with_columns(score.alias(f"score_{col_name}"))

    return result


def phase1_hard_filter(
    factor_df: pl.DataFrame,
    thresholds: dict | None = None,
) -> pl.DataFrame:
    """Phase 1 hard filter: MOM_5D>0, (VOL_ANOMALY=1 OR MFI_14>40), ATR_RATIO<0.8."""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.copy()

    mom_min = thresholds["mom_5d_min"]
    mfi_min = thresholds["mfi_14_min"]
    atr_max = thresholds["atr_ratio_max"]

    df = factor_df.filter(
        (pl.col("mom_5d") > mom_min)
        & ((pl.col("vol_anomaly") == 1) | (pl.col("mfi_14") > mfi_min))
        & (pl.col("atr_ratio") < atr_max)
    )
    return df


def phase2_score(factor_df: pl.DataFrame, weights: dict | None = None) -> pl.DataFrame:
    """Compute Coarse_Score = Σ(w_i × z_capped_i), sorted descending."""
    if weights is None:
        weights = FACTOR_WEIGHTS

    z_cols = [f"z_{name}" for name in weights if f"z_{name}" in factor_df.columns]
    if not z_cols:
        return factor_df.with_columns(pl.lit(0.0).alias("coarse_score"))

    expr = None
    for col_name in z_cols:
        factor_name = col_name[2:]  # strip "z_"
        w = weights.get(factor_name, 0.0)
        term = w * pl.col(col_name)
        if expr is None:
            expr = term
        else:
            expr = expr + term

    return factor_df.with_columns(expr.alias("coarse_score")).sort("coarse_score", descending=True)


def compute_missing_rate(factor_df: pl.DataFrame) -> pl.Series:
    """Compute per-row missing factor rate (fraction of NaN/None)."""
    factor_cols = [c for c in factor_df.columns if c not in ("ticker", "date")]
    if not factor_cols:
        return pl.Series("missing_rate", [0.0] * len(factor_df))

    n = len(factor_cols)
    null_count = None
    for col_name in factor_cols:
        col_null = factor_df[col_name].is_null().cast(pl.Float64)
        if null_count is None:
            null_count = col_null
        else:
            null_count = null_count + col_null

    assert null_count is not None
    return null_count / n


def dedup_by_sector_industry(
    df: pl.DataFrame,
    sector_cap: int,
    industry_cap: int,
    top_n: int = 20,
) -> pl.DataFrame:
    """Greedy dedup by sector/industry caps, sorted by coarse_score descending."""
    if isinstance(sector_cap, bool) or not isinstance(sector_cap, int) or sector_cap < 1:
        raise ValueError(f"sector_cap must be a positive integer, got {sector_cap!r}")
    if isinstance(industry_cap, bool) or not isinstance(industry_cap, int) or industry_cap < 1:
        raise ValueError(f"industry_cap must be a positive integer, got {industry_cap!r}")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError(f"top_n must be a positive integer, got {top_n!r}")
    for col in ("sector", "industry", "coarse_score"):
        if col not in df.columns:
            raise KeyError(f"DataFrame missing required column: {col!r}")

    work = df.sort("coarse_score", descending=True).with_columns(
        pl.col("sector").fill_null(MISSING_SECTOR_INDUSTRY),
        pl.col("industry").fill_null(MISSING_SECTOR_INDUSTRY),
    )

    sector_counts: dict[str, int] = {}
    industry_counts: dict[str, int] = {}
    keep_mask: list[bool] = []
    kept = 0

    for row in work.iter_rows(named=True):
        sector = row["sector"]
        industry = row["industry"]
        sc = sector_counts.get(sector, 0)
        ic = industry_counts.get(industry, 0)
        if sc < sector_cap and ic < industry_cap and kept < top_n:
            keep_mask.append(True)
            sector_counts[sector] = sc + 1
            industry_counts[industry] = ic + 1
            kept += 1
        else:
            keep_mask.append(False)
            if kept >= top_n:
                break

    return work[[i for i, ok in enumerate(keep_mask) if ok], :]


class DynamicThreshold:
    """Auto-adjust hard filter thresholds based on daily pass rate."""

    def __init__(self):
        self._last_adjustment: date | None = None
        self._cumulative_widen = 0.0
        self._thresholds = DEFAULT_THRESHOLDS.copy()

    @property
    def thresholds(self) -> dict:
        return self._thresholds.copy()

    @property
    def cooldown_days(self) -> int:
        return 3

    @property
    def max_cumulative(self) -> float:
        return 0.30

    @property
    def step_size(self) -> float:
        return 0.10

    def evaluate(self, pass_rate: float, today: date) -> Tuple[dict, str, str]:
        if pass_rate is None or pass_rate == 0:
            return self._thresholds.copy(), "unknown", "none"

        if PASS_RATE_NORMAL_LO <= pass_rate <= PASS_RATE_NORMAL_HI:
            status, action = "normal", "none"
        elif PASS_RATE_NORMAL_HI < pass_rate <= PASS_RATE_TIGHT:
            status, action = "tight", "warn"
        elif PASS_RATE_TIGHT < pass_rate <= PASS_RATE_VERY_TIGHT:
            status, action = "very_tight", "widen"
        elif pass_rate > PASS_RATE_VERY_TIGHT:
            status, action = "extreme", "extreme_widen"
        elif pass_rate < PASS_RATE_MIN:
            status, action = "loose", "tighten"
        else:
            status, action = "normal", "none"

        if action in ("widen", "extreme_widen"):
            self._try_widen(today)
        elif action == "tighten":
            self._try_tighten(today)

        return self._thresholds.copy(), status, action

    def _try_widen(self, today: date):
        if self._last_adjustment and (today - self._last_adjustment).days < self.cooldown_days:
            return
        if self._cumulative_widen >= self.max_cumulative:
            return

        step = self.step_size
        self._cumulative_widen = min(self._cumulative_widen + step, self.max_cumulative)

        self._thresholds["mom_5d_min"] -= step * abs(DEFAULT_THRESHOLDS["mom_5d_min"] + 0.01)
        self._thresholds["mfi_14_min"] -= step * abs(DEFAULT_THRESHOLDS["mfi_14_min"])
        self._thresholds["atr_ratio_max"] += step * abs(DEFAULT_THRESHOLDS["atr_ratio_max"])

        self._last_adjustment = today
        logger.info(
            "Dynamic threshold widened +%.0f%% (total +%.0f%%)",
            step * 100,
            self._cumulative_widen * 100,
        )

    def _try_tighten(self, today: date):
        if self._last_adjustment and (today - self._last_adjustment).days < self.cooldown_days:
            return
        if self._cumulative_widen <= 0:
            return

        step = self.step_size
        self._cumulative_widen = max(self._cumulative_widen - step, 0.0)

        self._thresholds["mom_5d_min"] += step * abs(DEFAULT_THRESHOLDS["mom_5d_min"] + 0.01)
        self._thresholds["mfi_14_min"] += step * abs(DEFAULT_THRESHOLDS["mfi_14_min"])
        self._thresholds["atr_ratio_max"] -= step * abs(DEFAULT_THRESHOLDS["atr_ratio_max"])

        self._last_adjustment = today
        logger.info(
            "Dynamic threshold tightened -%.0f%% (total +%.0f%%)",
            step * 100,
            self._cumulative_widen * 100,
        )
