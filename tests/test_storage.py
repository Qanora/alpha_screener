"""Tests for Parquet data storage."""

import tempfile
from datetime import date
from pathlib import Path

import polars as pl

from alphascreener.core.storage import DataStore


def _sample_df():
    return pl.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "date": [date(2026, 5, 7), date(2026, 5, 7)],
            "open": [289.0, 420.0],
            "high": [292.0, 428.0],
            "low": [285.0, 418.0],
            "close": [287.0, 421.0],
            "volume": [45_000_000, 35_000_000],
        }
    )


class TestDataStore:
    def test_init_custom_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DataStore(home=Path(tmp))
            assert store.home == Path(tmp)
            assert store.ohlcv_dir == Path(tmp) / "data" / "ohlcv"

    def test_write_and_read_ohlcv(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DataStore(home=Path(tmp))
            dt = date(2026, 5, 7)
            df = _sample_df()
            store.write_ohlcv(df, dt)
            result = store.read_ohlcv(dt)
            assert result is not None
            assert len(result) == 2

    def test_ohlcv_partition_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DataStore(home=Path(tmp))
            dt = date(2026, 5, 7)
            store.write_ohlcv(_sample_df(), dt)
            expected_dir = Path(tmp) / "data" / "ohlcv" / "dt=2026-05-07"
            assert expected_dir.exists()
            assert (expected_dir / "data.parquet").exists()

    def test_read_missing_ohlcv(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DataStore(home=Path(tmp))
            result = store.read_ohlcv(date(2020, 1, 1))
            assert result is None

    def test_write_signals_llm_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DataStore(home=Path(tmp))
            dt = date(2026, 5, 7)
            df = pl.DataFrame({"ticker": ["AAPL"], "signal": [0.85]})
            store.write_signals(df, dt, track="llm")
            path = Path(tmp) / "data" / "signals" / "dt=2026-05-07" / "signals_refined.parquet"
            assert path.exists()

    def test_write_signals_pure_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DataStore(home=Path(tmp))
            dt = date(2026, 5, 7)
            df = pl.DataFrame({"ticker": ["AAPL"], "signal": [0.82]})
            store.write_signals(df, dt, track="pure")
            path = Path(tmp) / "data" / "signals" / "dt=2026-05-07" / "signals_refined_pure.parquet"
            assert path.exists()

    def test_read_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DataStore(home=Path(tmp))
            dt = date(2026, 5, 7)
            df = pl.DataFrame({"ticker": ["AAPL"], "signal": [0.85]})
            store.write_signals(df, dt, track="llm")
            result = store.read_signals(dt, track="llm")
            assert result is not None
            assert result["ticker"][0] == "AAPL"

    def test_write_universe_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DataStore(home=Path(tmp))
            df = pl.DataFrame({"ticker": ["AAPL"], "sector": ["Technology"]})
            store.write_universe_meta(df)
            result = store.read_universe_meta()
            assert result is not None
            assert result["sector"][0] == "Technology"

    def test_read_universe_meta_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DataStore(home=Path(tmp))
            assert store.read_universe_meta() is None

    def test_write_backtest_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DataStore(home=Path(tmp))
            df = pl.DataFrame({"pnl": [0.05, -0.02]})
            store.write_backtest_results(df, 2026, 5)
            path = Path(tmp) / "data" / "backtest" / "dt=2026-05" / "backtest.parquet"
            assert path.exists()
