"""YFinance data adapter with rate limiting and retry."""

import asyncio
import logging
from datetime import date
from typing import List, Tuple

import pandas as pd
import polars as pl
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

OHLCV_COL_MAP = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
RETRY_MAX_ATTEMPTS = 3
RETRY_MIN_WAIT = 2
RETRY_MAX_WAIT = 60


class YFinanceAdapter:
    """Rate-limited yfinance client for batch OHLCV downloads."""

    def __init__(self, rps: int = 5):
        self._semaphore = asyncio.Semaphore(rps)

    @retry(
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    def fetch_ohlcv_batch(self, tickers: List[str], start: date, end: date) -> pl.DataFrame:
        """Download daily OHLCV for a batch of tickers with retry."""
        logger.info("Fetching %d tickers from %s to %s", len(tickers), start, end)
        if not tickers:
            return pl.DataFrame()
        try:
            df = yf.download(
                tickers=tickers,
                start=str(start),
                end=str(end),
                group_by="ticker",
                threads=False,
                progress=False,
                auto_adjust=True,
            )
        except Exception:
            logger.warning("yfinance download failed for %d tickers, retrying...", len(tickers))
            raise

        if df.empty:
            logger.warning("Empty response for batch of %d tickers", len(tickers))
            return pl.DataFrame(
                schema={
                    "ticker": pl.Utf8,
                    "date": pl.Date,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                }
            )

        if isinstance(df.columns, pd.MultiIndex):
            records = []
            for ticker in df.columns.get_level_values(0).unique():
                ticker_df = df[ticker]
                ticker_df["ticker"] = ticker
                ticker_df["date"] = ticker_df.index
                ticker_df = ticker_df.reset_index(drop=True)
                tdf = pl.from_pandas(ticker_df)
                tdf = tdf.rename({k: v for k, v in OHLCV_COL_MAP.items() if k in tdf.columns})
                records.append(tdf)
            return pl.concat(records)
        else:
            df["ticker"] = tickers[0]
            df["date"] = df.index
            result = pl.from_pandas(df.reset_index(drop=True))
            return result.rename({k: v for k, v in OHLCV_COL_MAP.items() if k in result.columns})

    async def fetch_all(
        self, tickers: List[str], start: date, end: date
    ) -> Tuple[pl.DataFrame, List[str]]:
        """Fetch all tickers in batches with rate limiting.

        Returns (full DataFrame, list of failed tickers).
        """
        all_data = []
        failed: List[str] = []
        batch_size = 50

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            async with self._semaphore:
                try:
                    batch_df = await asyncio.to_thread(self.fetch_ohlcv_batch, batch, start, end)
                    if not batch_df.is_empty():
                        all_data.append(batch_df)
                except Exception as e:
                    logger.warning(
                        "Batch %d/%d failed: %s",
                        i // batch_size + 1,
                        -(-len(tickers) // batch_size),
                        e,
                    )
                    failed.extend(batch)

        combined = pl.concat(all_data) if all_data else pl.DataFrame()
        return combined, failed

    def fetch_ticker_info(self, ticker: str) -> dict:
        """Fetch yfinance Ticker.info for a single ticker."""
        try:
            t = yf.Ticker(ticker)
            return t.info or {}
        except Exception:
            logger.debug("Failed to fetch info for %s", ticker)
            return {}

    def fetch_earnings_dates(self, ticker: str) -> List[date]:
        """Fetch recent earnings dates for PEAD_FLAG computation."""
        try:
            t = yf.Ticker(ticker)
            eds = t.earnings_dates
            if eds is None or eds.empty:
                return []
            return [d.date() for d in eds.index[:4]]
        except Exception:
            return []

    def fetch_earnings_history(self, ticker: str) -> dict:
        """Fetch reported EPS history."""
        try:
            t = yf.Ticker(ticker)
            hist = t.earnings_history
            if hist is None or hist.empty:
                return {}
            return hist.to_dict()
        except Exception:
            return {}
