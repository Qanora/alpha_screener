"""Stooq backup OHLCV adapter for cross-validation."""

import logging
from datetime import date, timedelta
from typing import List, Optional
from urllib.parse import urlencode

import polars as pl
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

STOOQ_BASE = "https://stooq.com/q/d/l/"


class StooqAdapter:
    """Stooq data source for OHLCV cross-validation."""

    def __init__(self, base_url: str = STOOQ_BASE):
        self.base_url = base_url.rstrip("/")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((requests.RequestException,)),
        reraise=True,
    )
    def fetch_ohlcv(self, ticker: str, start: date, end: date) -> Optional[pl.DataFrame]:
        """Fetch daily OHLCV for a single ticker from Stooq.

        Returns DataFrame with lowercase columns (date, open, high, low, close, volume)
        or None if not available.
        """
        params = {
            "s": ticker.lower().replace("-", "."),
            "d1": start.strftime("%Y%m%d"),
            "d2": end.strftime("%Y%m%d"),
            "i": "d",
        }
        url = f"{self.base_url}/?{urlencode(params)}"
        logger.debug("Stooq request: %s", url)

        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            logger.warning("Stooq returned %d for %s", resp.status_code, ticker)
            return None

        csv_text = resp.text.strip()
        if not csv_text or "No data" in csv_text:
            return None

        try:
            df = pl.read_csv(csv_text.encode(), has_header=True)
            if df.is_empty():
                return None

            expected_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
            if not expected_cols.issubset(set(df.columns)):
                return None

            df = df.rename(
                {
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            ).with_columns(
                [
                    pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("volume").cast(pl.Float64),
                ]
            )
            return df
        except Exception as e:
            logger.debug("Stooq parse error for %s: %s", ticker, e)
            return None

    def fetch_batch(self, tickers: List[str], start: date, end: date) -> dict:
        """Fetch Stooq OHLCV for multiple tickers.

        Returns {ticker: DataFrame | None}.
        """
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.fetch_ohlcv(ticker, start, end)
            except Exception as e:
                logger.debug("Stooq fetch error for %s: %s", ticker, e)
                results[ticker] = None
        return results


class CrossValidator:
    """Compare yfinance vs fallback OHLCV and record discrepancies."""

    _FIELDS = ["open", "high", "low", "close"]
    DIFF_THRESHOLD = 0.005

    def __init__(self, stooq: StooqAdapter):
        self.stooq = stooq

    def validate(
        self,
        yf_df: pl.DataFrame,
        tickers: List[str],
        metric_date: date,
    ) -> pl.DataFrame:
        """Cross-validate yfinance OHLCV against Stooq.

        Returns a DataFrame matching data_source_diff table schema.
        """
        stooq_data = self.stooq.fetch_batch(tickers, metric_date, metric_date + timedelta(days=1))
        diffs = []

        for ticker in tickers:
            st_df = stooq_data.get(ticker)
            if st_df is None or st_df.is_empty():
                continue

            yf_ticker = yf_df.filter(pl.col("ticker") == ticker)
            if yf_ticker.is_empty():
                continue

            for field in self._FIELDS:
                try:
                    yf_val = yf_ticker[field].mean()
                    st_val = st_df[field].mean()
                except Exception:
                    continue

                if yf_val is None or st_val is None or st_val == 0:
                    continue
                if not isinstance(yf_val, (int, float)) or not isinstance(st_val, (int, float)):
                    continue

                diff_pct = abs(yf_val - st_val) / abs(st_val)
                if diff_pct > self.DIFF_THRESHOLD:
                    diffs.append(
                        {
                            "metric_date": metric_date,
                            "ticker": ticker,
                            "field": field,
                            "yfinance_value": float(yf_val),
                            "fallback_value": float(st_val),
                            "fallback_source": "stooq",
                            "diff_pct": round(diff_pct, 6),
                            "alerted": False,
                        }
                    )

        return pl.DataFrame(diffs) if diffs else pl.DataFrame()
