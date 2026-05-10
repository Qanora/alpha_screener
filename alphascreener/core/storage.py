"""Parquet storage for OHLCV, factors, and signals data."""

import logging
from datetime import date
from pathlib import Path
from typing import Literal, Optional

import polars as pl

from alphascreener.config import get_settings

logger = logging.getLogger(__name__)

SignalTrack = Literal["llm", "pure"]


class DataStore:
    """Manages Parquet storage for all data layers."""

    def __init__(self, home: Optional[Path] = None):
        if home is None:
            home = get_settings().home
        self.home = home
        self.ohlcv_dir = home / "data" / "ohlcv"
        self.factors_dir = home / "data" / "factors"
        self.signals_dir = home / "data" / "signals"
        self.backtest_dir = home / "data" / "backtest"
        self.case_library_dir = home / "data" / "case_library"

    def _partition_path(self, base: Path, dt: date) -> Path:
        return base / f"dt={dt.isoformat()}"

    def write_ohlcv(self, df: pl.DataFrame, dt: date) -> Path:
        path = self._partition_path(self.ohlcv_dir, dt)
        path.mkdir(parents=True, exist_ok=True)
        out = path / "data.parquet"
        df.write_parquet(str(out))
        logger.info("Wrote %d rows to %s", len(df), out)
        return out

    def read_ohlcv(self, dt: date) -> Optional[pl.DataFrame]:
        path = self._partition_path(self.ohlcv_dir, dt) / "data.parquet"
        if not path.exists():
            return None
        return pl.read_parquet(str(path))

    def write_factors(self, df: pl.DataFrame, dt: date) -> Path:
        path = self._partition_path(self.factors_dir, dt)
        path.mkdir(parents=True, exist_ok=True)
        out = path / "factors.parquet"
        df.write_parquet(str(out))
        return out

    def write_signals(self, df: pl.DataFrame, dt: date, track: SignalTrack = "llm") -> Path:
        path = self._partition_path(self.signals_dir, dt)
        path.mkdir(parents=True, exist_ok=True)
        filename = "signals_refined.parquet" if track == "llm" else "signals_refined_pure.parquet"
        out = path / filename
        df.write_parquet(str(out))
        return out

    def read_signals(self, dt: date, track: SignalTrack = "llm") -> Optional[pl.DataFrame]:
        filename = "signals_refined.parquet" if track == "llm" else "signals_refined_pure.parquet"
        path = self._partition_path(self.signals_dir, dt) / filename
        if not path.exists():
            return None
        return pl.read_parquet(str(path))

    def write_universe_meta(self, df: pl.DataFrame) -> Path:
        out = self.home / "data" / "universe_meta.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(str(out))
        return out

    def read_universe_meta(self) -> Optional[pl.DataFrame]:
        path = self.home / "data" / "universe_meta.parquet"
        if not path.exists():
            return None
        return pl.read_parquet(str(path))

    def write_backtest_results(self, df: pl.DataFrame, year: int, month: int) -> Path:
        path = self.backtest_dir / f"dt={year:04d}-{month:02d}"
        path.mkdir(parents=True, exist_ok=True)
        out = path / "backtest.parquet"
        df.write_parquet(str(out))
        return out
