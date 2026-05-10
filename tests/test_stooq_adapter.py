"""Tests for Stooq backup adapter and CrossValidator."""

from datetime import date, timedelta

import polars as pl

from alphascreener.adapters.stooq_adapter import STOOQ_BASE, CrossValidator, StooqAdapter


class TestStooqAdapter:
    def test_init_default_url(self):
        adapter = StooqAdapter()
        assert adapter.base_url == STOOQ_BASE.rstrip("/")

    def test_init_custom_url(self):
        adapter = StooqAdapter(base_url="https://example.com/data/")
        assert adapter.base_url == "https://example.com/data"

    def test_fetch_ohlcv_returns_none_or_dataframe(self):
        adapter = StooqAdapter()
        today = date.today()
        result = adapter.fetch_ohlcv("AAPL", today - timedelta(days=5), today)
        assert result is None or isinstance(result, pl.DataFrame)

    def test_fetch_batch_returns_dict(self):
        adapter = StooqAdapter()
        today = date.today()
        results = adapter.fetch_batch(["AAPL", "MSFT"], today - timedelta(days=5), today)
        assert isinstance(results, dict)
        assert "AAPL" in results
        assert "MSFT" in results


class TestCrossValidator:
    def test_init(self):
        stooq = StooqAdapter()
        validator = CrossValidator(stooq)
        assert validator.stooq is stooq
        assert validator.DIFF_THRESHOLD == 0.005

    def test_validate_empty_yf_df(self):
        stooq = StooqAdapter()
        validator = CrossValidator(stooq)
        today = date.today()
        empty_df = pl.DataFrame(
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
        result = validator.validate(empty_df, ["AAPL"], today)
        assert result.is_empty()

    def test_validate_no_stooq_data(self):
        stooq = StooqAdapter()
        validator = CrossValidator(stooq)
        today = date.today()
        yf_df = pl.DataFrame(
            {
                "ticker": ["ZZZZZ_INVALID"],
                "date": [today],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1_000_000],
            }
        )
        result = validator.validate(yf_df, ["ZZZZZ_INVALID"], today)
        # Should return empty — Stooq likely has no data for invalid ticker
        assert isinstance(result, pl.DataFrame)

    def test_validate_identical_data(self):
        """When yfinance and Stooq data match, diff should be empty."""

        class FakeStooq(StooqAdapter):
            def fetch_batch(self, tickers, start, end):
                today = date.today()
                return {
                    t: pl.DataFrame(
                        {
                            "date": [today],
                            "open": [100.0],
                            "high": [102.0],
                            "low": [99.0],
                            "close": [101.0],
                            "volume": [5_000_000],
                        }
                    )
                    for t in tickers
                }

        validator = CrossValidator(FakeStooq())
        today = date.today()
        yf_df = pl.DataFrame(
            {
                "ticker": ["TEST"],
                "date": [today],
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                "volume": [5_000_000],
            }
        )
        result = validator.validate(yf_df, ["TEST"], today)
        assert result.is_empty()

    def test_validate_divergent_data(self):
        """When yfinance and Stooq differ beyond threshold, diffs should appear."""

        class FakeStooq(StooqAdapter):
            def fetch_batch(self, tickers, start, end):
                today = date.today()
                return {
                    t: pl.DataFrame(
                        {
                            "date": [today],
                            "open": [100.0],
                            "high": [102.0],
                            "low": [99.0],
                            "close": [105.0],  # different!
                            "volume": [5_000_000],
                        }
                    )
                    for t in tickers
                }

        validator = CrossValidator(FakeStooq())
        today = date.today()
        yf_df = pl.DataFrame(
            {
                "ticker": ["TEST"],
                "date": [today],
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],  # 3.8% diff vs Stooq's 105
                "volume": [5_000_000],
            }
        )
        result = validator.validate(yf_df, ["TEST"], today)
        assert not result.is_empty()
        assert result["field"][0] == "close"
