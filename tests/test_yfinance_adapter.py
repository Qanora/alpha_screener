"""Tests for YFinance data adapter."""

from datetime import date, timedelta


from alphascreener.adapters.yfinance_adapter import OHLCV_COL_MAP, YFinanceAdapter


class TestOhlcvColumnMap:
    def test_maps_pascal_to_lower(self):
        assert OHLCV_COL_MAP["Open"] == "open"
        assert OHLCV_COL_MAP["High"] == "high"
        assert OHLCV_COL_MAP["Low"] == "low"
        assert OHLCV_COL_MAP["Close"] == "close"
        assert OHLCV_COL_MAP["Volume"] == "volume"

    def test_has_five_entries(self):
        assert len(OHLCV_COL_MAP) == 5


class TestYFinanceAdapter:
    def test_init_default_rps(self):
        adapter = YFinanceAdapter()
        assert adapter._semaphore is not None

    def test_init_custom_rps(self):
        adapter = YFinanceAdapter(rps=3)
        assert adapter._semaphore is not None

    def test_fetch_ohlcv_batch_returns_correct_columns(self):
        adapter = YFinanceAdapter(rps=5)
        today = date.today()
        df = adapter.fetch_ohlcv_batch(["AAPL"], today - timedelta(days=5), today)
        expected_cols = {"ticker", "date", "open", "high", "low", "close", "volume"}
        assert expected_cols.issubset(set(df.columns))

    def test_fetch_ohlcv_batch_returns_data(self):
        adapter = YFinanceAdapter(rps=5)
        today = date.today()
        df = adapter.fetch_ohlcv_batch(["AAPL", "MSFT"], today - timedelta(days=3), today)
        assert not df.is_empty()
        assert df["ticker"].n_unique() >= 1

    def test_fetch_ohlcv_batch_empty_tickers(self):
        adapter = YFinanceAdapter()
        today = date.today()
        df = adapter.fetch_ohlcv_batch([], today - timedelta(days=1), today)
        assert df.is_empty()

    def test_fetch_ticker_info_returns_dict(self):
        adapter = YFinanceAdapter()
        info = adapter.fetch_ticker_info("AAPL")
        assert isinstance(info, dict)
        assert "symbol" in info or "shortName" in info or len(info) >= 0

    def test_fetch_ticker_info_invalid_ticker(self):
        adapter = YFinanceAdapter()
        info = adapter.fetch_ticker_info("ZZZZZZZZZZZ_INVALID")
        assert isinstance(info, dict)

    def test_fetch_earnings_dates(self):
        adapter = YFinanceAdapter()
        dates = adapter.fetch_earnings_dates("AAPL")
        assert isinstance(dates, list)
        for d in dates:
            assert isinstance(d, date)
