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
        z_capped = z.clip(-3.0, 3.0)
        score = 50.0 + z_capped * (50.0 / 3.0)
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

    return null_count / n


class DynamicThreshold:
    """Auto-adjust hard filter thresholds based on daily pass rate."""

    def __init__(self):
        self._last_adjustment: date | None = None
        self._cumulative_widen = 0.0
        self._thresholds = DEFAULT_THRESHOLDS.copy()

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

        if 0.80 <= pass_rate <= 0.92:
            status, action = "normal", "none"
        elif 0.92 < pass_rate <= 0.95:
            status, action = "tight", "warn"
        elif 0.95 < pass_rate <= 0.98:
            status, action = "very_tight", "widen"
        elif pass_rate > 0.98:
            status, action = "extreme", "extreme_widen"
        elif pass_rate < 0.70:
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
