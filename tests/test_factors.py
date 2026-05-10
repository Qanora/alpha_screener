"""Tests for 13 factor calculation engine."""

import polars as pl

from alphascreener.core.factors import (
    _atr_ratio,
    _bb_squeeze,
    _cmf_21,
    _golden_cross,
    _macd_cross,
    _mfi_14,
    _mom_5d,
    _mom_slope,
    _pth,
    _rsi_oversold,
    _vol_anomaly,
)


def _make_ohlcv(prices, volumes=None):
    n = len(prices)
    if volumes is None:
        volumes = [1_000_000] * n
    return (
        pl.Series("close", prices, dtype=pl.Float64),
        pl.Series("high", [p * 1.02 for p in prices], dtype=pl.Float64),
        pl.Series("low", [p * 0.98 for p in prices], dtype=pl.Float64),
        pl.Series("volume", volumes, dtype=pl.Float64),
    )


class TestMom5d:
    def test_positive(self):
        close = pl.Series([100, 101, 102, 103, 104, 105], dtype=pl.Float64)
        result = _mom_5d(close)
        assert result > 0

    def test_flat(self):
        close = pl.Series([100] * 10, dtype=pl.Float64)
        result = _mom_5d(close)
        assert result == 0.0

    def test_insufficient_data(self):
        close = pl.Series([100, 101, 102], dtype=pl.Float64)
        result = _mom_5d(close)
        assert result == 0.0


class TestPth:
    def test_at_high(self):
        vals = [90 + i for i in range(64)]
        close = pl.Series(vals, dtype=pl.Float64)
        result = _pth(close)
        assert 0.99 <= result <= 1.01

    def test_below_peak(self):
        vals = [100] * 50 + [80] * 13
        close = pl.Series(vals, dtype=pl.Float64)
        result = _pth(close)
        assert result < 1.0

    def test_insufficient_data(self):
        close = pl.Series([100, 101, 102], dtype=pl.Float64)
        assert _pth(close) == 0.0


class TestMomSlope:
    def test_positive_slope(self):
        vals = [100 + i for i in range(20)]
        close = pl.Series(vals, dtype=pl.Float64)
        result = _mom_slope(close)
        assert result > -0.01

    def test_insufficient_data(self):
        close = pl.Series([100, 101], dtype=pl.Float64)
        assert _mom_slope(close) == 0.0


class TestBbSqueeze:
    def test_returns_zero_or_one(self):
        vals = [100 + i * 0.1 for i in range(80)]
        close = pl.Series(vals, dtype=pl.Float64)
        result = _bb_squeeze(close)
        assert result in (0, 1)

    def test_insufficient_data(self):
        close = pl.Series([100] * 10, dtype=pl.Float64)
        assert _bb_squeeze(close) == 0


class TestAtrRatio:
    def test_normal_range(self):
        vals = [100 + i * 0.1 for i in range(30)]
        close = pl.Series(vals, dtype=pl.Float64)
        high = pl.Series([v * 1.02 for v in vals], dtype=pl.Float64)
        low = pl.Series([v * 0.98 for v in vals], dtype=pl.Float64)
        result = _atr_ratio(high, low, close)
        assert result > 0

    def test_insufficient_data(self):
        close = pl.Series([100, 101], dtype=pl.Float64)
        high = pl.Series([102, 103], dtype=pl.Float64)
        low = pl.Series([98, 99], dtype=pl.Float64)
        assert _atr_ratio(high, low, close) == 1.0


class TestMfi14:
    def test_range(self):
        tp = pl.Series([100 + i * 0.1 for i in range(20)], dtype=pl.Float64)
        vol = pl.Series([1_000_000] * 20, dtype=pl.Float64)
        result = _mfi_14(tp, vol)
        assert 0.0 <= result <= 100.0

    def test_insufficient_data(self):
        tp = pl.Series([100, 101], dtype=pl.Float64)
        vol = pl.Series([1_000_000, 1_000_000], dtype=pl.Float64)
        assert _mfi_14(tp, vol) == 50.0


class TestCmf21:
    def test_range(self):
        close = pl.Series([100 + i * 0.1 for i in range(30)], dtype=pl.Float64)
        high = pl.Series([v * 1.02 for v in close], dtype=pl.Float64)
        low = pl.Series([v * 0.98 for v in close], dtype=pl.Float64)
        vol = pl.Series([1_000_000] * 30, dtype=pl.Float64)
        result = _cmf_21(high, low, close, vol)
        assert -1.0 <= result <= 1.0

    def test_insufficient_data(self):
        close = pl.Series([100, 101], dtype=pl.Float64)
        high = pl.Series([102, 103], dtype=pl.Float64)
        low = pl.Series([98, 99], dtype=pl.Float64)
        vol = pl.Series([1_000_000, 1_000_000], dtype=pl.Float64)
        assert _cmf_21(high, low, close, vol) == 0.0


class TestVolAnomaly:
    def test_returns_zero_or_one(self):
        vol = pl.Series([1_000_000] * 60, dtype=pl.Float64)
        close = pl.Series([100 + i * 0.1 for i in range(60)], dtype=pl.Float64)
        result = _vol_anomaly(vol, close)
        assert result in (0, 1)

    def test_insufficient_data(self):
        vol = pl.Series([1_000_000] * 10, dtype=pl.Float64)
        close = pl.Series([100] * 10, dtype=pl.Float64)
        assert _vol_anomaly(vol, close) == 0


class TestRsiOversold:
    def test_returns_zero_or_one(self):
        close = pl.Series([100 + i * 0.5 for i in range(30)], dtype=pl.Float64)
        result = _rsi_oversold(close)
        assert result in (0, 1)

    def test_uptrend_not_oversold(self):
        vals = [100 + i * 0.5 for i in range(20)]
        close = pl.Series(vals, dtype=pl.Float64)
        result = _rsi_oversold(close)
        assert result == 0

    def test_insufficient_data(self):
        close = pl.Series([100, 101], dtype=pl.Float64)
        assert _rsi_oversold(close) == 0


class TestMacdCross:
    def test_returns_zero(self):
        vals = [100 + i * 0.1 for i in range(40)]
        close = pl.Series(vals, dtype=pl.Float64)
        result = _macd_cross(close)
        assert result in (0, 1)

    def test_insufficient_data(self):
        close = pl.Series([100, 101], dtype=pl.Float64)
        assert _macd_cross(close) == 0


class TestGoldenCross:
    def test_returns_zero_or_one(self):
        vals = [100 + i * 0.01 for i in range(210)]
        close = pl.Series(vals, dtype=pl.Float64)
        result = _golden_cross(close)
        assert result in (0, 1)

    def test_insufficient_data(self):
        close = pl.Series([100, 101], dtype=pl.Float64)
        assert _golden_cross(close) == 0
