"""13-factor calculation engine.

All factors operate on polars DataFrames grouped by ticker.
Each factor returns a DataFrame with (ticker, date, factor_name value).
"""

import logging
from datetime import date, timedelta
from typing import List

import polars as pl

from alphascreener.adapters.yfinance_adapter import YFinanceAdapter

logger = logging.getLogger(__name__)

MAX_LOOKBACK = 210
MOM_SLOPE_WINDOW = 10
PTH_WINDOW = 63
BB_WINDOW = 60
ATR_SHORT = 5
ATR_LONG = 20
MFI_WINDOW = 14
CMF_WINDOW = 21
VOL_Z_WINDOW = 50
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
INSIDER_WINDOW = 60


def _rolling_slope(returns: pl.Series, window: int) -> pl.Series:
    """Linear regression slope of returns over a rolling window."""
    n = window
    x = pl.arange(0, n, eager=True).cast(pl.Float64)
    sum_x = x.sum()
    sum_x2 = (x * x).sum()
    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        return pl.Series([0.0] * len(returns))

    slopes = []
    vals = returns.to_list()
    for i in range(len(vals)):
        if i < n - 1:
            slopes.append(0.0)
        else:
            y = pl.Series(vals[i - n + 1 : i + 1])
            sum_y = y.sum()
            sum_xy = (x * y).sum()
            slope = (n * sum_xy - sum_x * sum_y) / denom
            slopes.append(float(slope))
    return pl.Series(slopes)


def compute_all_factors(
    ohlcv: pl.DataFrame, tickers: List[str], yf_adapter: YFinanceAdapter
) -> pl.DataFrame:
    """Compute all 13 factors for given tickers from OHLCV data.

    ohlcv: DataFrame with columns [ticker, date, open, high, low, close, volume]
    Returns: DataFrame with columns [ticker, date] + 13 factor columns
    """
    if ohlcv.is_empty():
        return pl.DataFrame()

    ohlcv = ohlcv.sort(["ticker", "date"])

    results = []
    for ticker in tickers:
        tdf = ohlcv.filter(pl.col("ticker") == ticker)
        if len(tdf) < 5:
            continue

        close = tdf["close"]
        high = tdf["high"]
        low = tdf["low"]
        volume = tdf["volume"]
        typical_price = (high + low + close) / 3

        factor_values = {"ticker": ticker, "date": tdf["date"].last()}

        factor_values["mom_5d"] = _mom_5d(close)
        factor_values["pth"] = _pth(close)
        factor_values["mom_slope"] = _mom_slope(close)
        factor_values["bb_squeeze"] = _bb_squeeze(close)
        factor_values["atr_ratio"] = _atr_ratio(high, low, close)
        factor_values["mfi_14"] = _mfi_14(typical_price, volume)
        factor_values["cmf_21"] = _cmf_21(high, low, close, volume)
        factor_values["vol_anomaly"] = _vol_anomaly(volume, close)
        factor_values["rsi_oversold"] = _rsi_oversold(close)
        factor_values["macd_cross"] = _macd_cross(close)
        factor_values["golden_cross"] = _golden_cross(close)
        ticker_info = yf_adapter.fetch_ticker_info(ticker)

        factor_values["pead_flag"] = _pead_flag(ticker, yf_adapter)
        factor_values["insider_buy"] = _insider_buy(ticker_info)
        factor_values["rev_accel"] = _rev_accel(ticker_info)

        results.append(factor_values)

    if not results:
        return pl.DataFrame()
    return pl.DataFrame(results)


def _mom_5d(close: pl.Series) -> float:
    if len(close) < 6:
        return 0.0
    return (close[-1] - close[-6]) / (close[-6] + 1e-8)


def _pth(close: pl.Series) -> float:
    if len(close) < PTH_WINDOW:
        return 0.0
    peak = close.tail(PTH_WINDOW).max()
    if peak == 0:
        return 0.0
    return close[-1] / peak


def _mom_slope(close: pl.Series) -> float:
    if len(close) < MOM_SLOPE_WINDOW + 1:
        return 0.0
    returns = (
        close.tail(MOM_SLOPE_WINDOW + 1).diff().drop_nulls()
        / close.tail(MOM_SLOPE_WINDOW + 1).shift(1).drop_nulls()
    )
    slopes = _rolling_slope(returns, MOM_SLOPE_WINDOW)
    return slopes[-1] if len(slopes) > 0 else 0.0


def _bb_squeeze(close: pl.Series) -> int:
    if len(close) < BB_WINDOW:
        return 0
    window = close.tail(BB_WINDOW)
    sma = window.mean()
    std = window.std()
    if std == 0:
        return 0
    bb_width = 4 * std / sma
    bb_20th = _rolling_bb_width_percentile(close, BB_WINDOW)
    return 1 if bb_width < bb_20th else 0


def _rolling_bb_width_percentile(close: pl.Series, window: int) -> float:
    """Compute the 20th percentile of BB Width over the window."""
    widths = []
    vals = close.to_list()
    for i in range(window - 1, len(vals)):
        w = vals[i - window + 1 : i + 1]
        sma = sum(w) / window
        std = (sum((x - sma) ** 2 for x in w) / window) ** 0.5
        widths.append(4 * std / sma if sma != 0 else 0)
    if not widths:
        return 0.0
    s = pl.Series(sorted(widths))
    idx = int(len(s) * 0.2)
    return s[idx]


def _atr_ratio(high: pl.Series, low: pl.Series, close: pl.Series) -> float:
    if len(close) < ATR_LONG + 1:
        return 1.0
    tr = pl.DataFrame({"h": high, "l": low, "c": close.shift(1)}).with_columns(
        pl.max_horizontal(
            pl.col("h") - pl.col("l"),
            (pl.col("h") - pl.col("c")).abs(),
            (pl.col("l") - pl.col("c")).abs(),
        ).alias("tr")
    )["tr"]
    atr5 = tr.tail(ATR_SHORT).mean()
    atr20 = tr.tail(ATR_LONG).mean()
    if atr5 is None or atr20 is None or atr20 == 0:
        return 1.0
    return float(atr5 / atr20)  # type: ignore[arg-type]


def _mfi_14(typical_price: pl.Series, volume: pl.Series) -> float:
    if len(typical_price) < MFI_WINDOW + 1:
        return 50.0
    mf = typical_price * volume
    tp_shifted = typical_price.shift(1)
    up = typical_price > tp_shifted
    down = typical_price < tp_shifted
    pos_sum = (up * mf).tail(MFI_WINDOW).sum()
    neg_sum = (down * mf).tail(MFI_WINDOW).sum()
    if neg_sum == 0:
        return 100.0
    ratio = pos_sum / (neg_sum + 1e-8)
    return 100.0 - 100.0 / (1.0 + ratio)


def _cmf_21(high: pl.Series, low: pl.Series, close: pl.Series, volume: pl.Series) -> float:
    if len(close) < CMF_WINDOW + 1:
        return 0.0
    mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-8)
    mf_volume = mf_multiplier * volume
    mf_sum = mf_volume.tail(CMF_WINDOW).sum()
    vol_sum = volume.tail(CMF_WINDOW).sum()
    if vol_sum is None or vol_sum == 0:
        return 0.0
    return float(mf_sum / vol_sum)


def _vol_anomaly(volume: pl.Series, close: pl.Series) -> int:
    if len(volume) < VOL_Z_WINDOW:
        return 0
    vol_tail = volume.tail(VOL_Z_WINDOW)
    mean = vol_tail.mean()
    std = vol_tail.std()
    if std is None or std == 0:
        return 0
    z = (volume[-1] - mean) / std
    sma5 = close.tail(5).mean()
    return 1 if z > 2.0 and close[-1] > sma5 else 0


def _rsi_oversold(close: pl.Series) -> int:
    if len(close) < RSI_WINDOW + 1:
        return 0
    delta = close.diff()
    gain = delta.clip(lower_bound=0)
    loss = (-delta).clip(lower_bound=0)
    avg_gain = gain.tail(RSI_WINDOW).mean()
    avg_loss = loss.tail(RSI_WINDOW).mean()
    if avg_gain is None or avg_loss is None or avg_loss == 0:
        rsi = 100.0
    else:
        rs = float(avg_gain / avg_loss)  # type: ignore[arg-type]
        rsi = 100.0 - 100.0 / (1.0 + rs)
    sma20 = close.tail(20).mean()
    return 1 if rsi < 30 and close[-1] > sma20 else 0


def _macd_cross(close: pl.Series) -> int:
    if len(close) < MACD_SLOW + MACD_SIGNAL:
        return 0
    ema12 = close.ewm_mean(span=MACD_FAST, adjust=False)
    ema26 = close.ewm_mean(span=MACD_SLOW, adjust=False)
    macd = ema12 - ema26
    signal = macd.ewm_mean(span=MACD_SIGNAL, adjust=False)
    hist = macd - signal
    if len(hist) < 2:
        return 0
    return 1 if macd[-1] > signal[-1] and hist[-1] > 0 and hist[-2] <= 0 else 0


def _golden_cross(close: pl.Series) -> int:
    if len(close) < 201:
        return 0
    sma50 = close.tail(50).mean()
    sma200 = close.tail(200).mean()
    sma50_prev = close.slice(-51, 50).mean()
    sma200_prev = close.slice(-201, 200).mean()
    return 1 if sma50 > sma200 and sma50_prev <= sma200_prev else 0


def _pead_flag(ticker: str, adapter: YFinanceAdapter) -> int:
    earnings_dates = adapter.fetch_earnings_dates(ticker)
    if not earnings_dates:
        return 0
    today = date.today()
    cutoff = today - timedelta(days=30)
    return 1 if any(cutoff <= d <= today for d in earnings_dates) else 0


def _insider_buy(info: dict) -> float:
    insider_pct = info.get("insiderPercentHeld") or info.get("heldPercentInsiders")
    if insider_pct is None:
        return 0.0
    mc = info.get("marketCap")
    if mc is None or mc == 0:
        return 0.0
    return float(insider_pct)


def _rev_accel(info: dict) -> float:
    q_growth = info.get("revenueGrowth")
    if q_growth is None:
        return 0.0
    return float(q_growth)
