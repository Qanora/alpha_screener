"""Universe management: index constituents and pre-filtering."""

import logging
from datetime import date, timedelta
from typing import List

import polars as pl

from alphascreener.config import get_settings

logger = logging.getLogger(__name__)

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
RUSSELL_1000_URL = "https://en.wikipedia.org/wiki/Russell_1000_Index"


def fetch_sp500_tickers() -> List[str]:
    """Scrape SP500 constituents from Wikipedia."""
    try:
        tables = pl.read_html(SP500_WIKI_URL)  # type: ignore[attr-defined]
        tickers = tables[0]["Symbol"].to_list()
        return [t.replace(".", "-") for t in tickers if t]
    except Exception as e:
        logger.error("Failed to fetch SP500 constituents: %s", e)
        return []


def fetch_russell1000_tickers() -> List[str]:
    """Scrape Russell 1000 constituents from Wikipedia."""
    try:
        tables = pl.read_html(RUSSELL_1000_URL)  # type: ignore[attr-defined]
        for table in tables:
            if "Ticker" in table.columns:
                tickers = table["Ticker"].to_list()
                return [t.replace(".", "-") for t in tickers if t]
        return []
    except Exception as e:
        logger.error("Failed to fetch Russell 1000 constituents: %s", e)
        return []


def fetch_index_universe() -> List[str]:
    """Fetch SP500 U Russell 1000 combined ticker list, deduplicated."""
    sp500 = fetch_sp500_tickers()
    russell = fetch_russell1000_tickers()
    combined = list(dict.fromkeys(sp500 + russell))
    logger.info(
        "Index universe: %d SP500 + %d Russell 1000 = %d unique",
        len(sp500),
        len(russell),
        len(combined),
    )
    return combined


def pre_filter(
    tickers: List[str],
    min_market_cap: float = 300_000_000,
    min_avg_volume: float = 20_000_000,
    min_price: float = 5.0,
    min_listing_months: int = 12,
) -> pl.DataFrame:
    """Filter tickers by static criteria using yfinance info.

    Does NOT import yfinance at module level to keep this callable
    from code that uses the adapter pattern.
    """
    from alphascreener.adapters.yfinance_adapter import YFinanceAdapter

    settings = get_settings()
    adapter = YFinanceAdapter(rps=settings.llm_rps)

    results = []
    today = date.today()
    min_listing_date = today - timedelta(days=min_listing_months * 30)

    for ticker in tickers:
        info = adapter.fetch_ticker_info(ticker)
        if not info:
            continue

        try:
            market_cap = info.get("marketCap")
            if market_cap is None or market_cap < min_market_cap:
                continue

            avg_volume = info.get("averageVolume")
            if avg_volume is None:
                avg_volume = info.get("averageDailyVolume10Day", 0)
            if avg_volume is None or avg_volume == 0:
                continue
            avg_volume_usd = float(avg_volume) * float(info.get("regularMarketPreviousClose", 0))
            if avg_volume_usd < min_avg_volume:
                continue

            prev_close = info.get("regularMarketPreviousClose")
            if prev_close is None or prev_close < min_price:
                continue

            quote_type = info.get("quoteType", "").upper()
            if quote_type not in ("EQUITY", "", None):
                continue

            ipo_date = info.get("ipoDate") or info.get("firstTradeDateEpochUtc")
            if ipo_date:
                if isinstance(ipo_date, (int, float)):
                    ipo = date.fromtimestamp(ipo_date)
                elif isinstance(ipo_date, str):
                    ipo = date.fromisoformat(ipo_date.split("T")[0])
                else:
                    ipo = None
                if ipo and ipo > min_listing_date:
                    continue

            sector = info.get("sector") or ""
            industry = info.get("industry") or ""

            results.append(
                {
                    "ticker": ticker,
                    "market_cap": float(market_cap) if market_cap else None,
                    "avg_volume_20d": float(avg_volume) if avg_volume else None,
                    "price": float(prev_close) if prev_close else None,
                    "sector": sector,
                    "industry": industry,
                    "listing_date": ipo_date,
                }
            )
        except Exception as e:
            logger.debug("Pre-filter skip %s: %s", ticker, e)
            continue

    df = pl.DataFrame(results) if results else pl.DataFrame()
    logger.info("Pre-filter: %d tickers passed from %d candidates", len(results), len(tickers))
    return df
