"""SQLite database initialization and all DDL."""

import contextlib
import sqlite3
from pathlib import Path


DDL = """
CREATE TABLE IF NOT EXISTS factor_versions (
    version TEXT PRIMARY KEY,
    released_at TIMESTAMP NOT NULL,
    config_json TEXT NOT NULL,
    parent_version TEXT,
    release_type TEXT CHECK(release_type IN ('MAJOR','MINOR','PATCH'))
);

CREATE TABLE IF NOT EXISTS paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_date DATE NOT NULL,
    ticker TEXT NOT NULL,
    rating TEXT NOT NULL,
    breakout_probability REAL NOT NULL,
    entry_price REAL,
    exit_price REAL,
    exit_reason TEXT,
    pnl_pct REAL,
    factor_version TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (factor_version) REFERENCES factor_versions(version)
);
CREATE INDEX IF NOT EXISTS idx_paper_trades_signal_date ON paper_trades(signal_date);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    triggered_at TIMESTAMP NOT NULL,
    severity TEXT CHECK(severity IN ('warning','critical')),
    rule_name TEXT NOT NULL,
    metric_value REAL,
    notes TEXT,
    resolved_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS llm_cost_daily (
    cost_date DATE PRIMARY KEY,
    total_usd REAL NOT NULL,
    call_count INTEGER NOT NULL,
    by_module_json TEXT
);

CREATE TABLE IF NOT EXISTS pid_lock (
    lock_name TEXT PRIMARY KEY,
    pid INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    acquired_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    meta_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_pid_lock_expires ON pid_lock(expires_at);

CREATE TABLE IF NOT EXISTS monitoring_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    sampled_at TIMESTAMP NOT NULL,
    rss_mb REAL NOT NULL,
    cpu_percent REAL NOT NULL,
    open_fd_count INTEGER,
    thread_count INTEGER,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_monitoring_task_time ON monitoring_samples(task_id, sampled_at);

CREATE TABLE IF NOT EXISTS alpha_acceptance_daily (
    metric_date DATE PRIMARY KEY,
    base_rate REAL NOT NULL,
    precision_at_20_pure REAL, precision_at_20_llm REAL,
    precision_at_10_pure REAL, precision_at_10_llm REAL,
    lift_at_20_pure REAL, lift_at_20_llm REAL,
    ic_pure REAL, ic_llm REAL,
    bootstrap_ci_lower_pure REAL, bootstrap_ci_upper_pure REAL,
    bootstrap_ci_lower_llm REAL, bootstrap_ci_upper_llm REAL,
    sample_size INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS data_source_diff (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_date DATE NOT NULL,
    ticker TEXT NOT NULL,
    field TEXT NOT NULL CHECK(field IN ('open','high','low','close','volume')),
    yfinance_value REAL NOT NULL,
    fallback_value REAL NOT NULL,
    fallback_source TEXT NOT NULL CHECK(fallback_source IN ('stooq','alpaca','polygon')),
    diff_pct REAL NOT NULL,
    alerted BOOLEAN DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_data_source_diff_date ON data_source_diff(metric_date);

CREATE TABLE IF NOT EXISTS factor_health_daily (
    metric_date DATE NOT NULL,
    factor_name TEXT NOT NULL,
    daily_ic REAL,
    rolling_ic_mean_90d REAL,
    cusum_value REAL,
    cusum_alert BOOLEAN DEFAULT 0,
    consecutive_alerts INTEGER DEFAULT 0,
    PRIMARY KEY (metric_date, factor_name)
);
CREATE INDEX IF NOT EXISTS idx_factor_health_factor_date ON factor_health_daily(factor_name, metric_date);
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    """Create database and all tables, enabling WAL mode."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(DDL)
    conn.commit()
    return conn


@contextlib.contextmanager
def get_db(db_path: Path):
    """Context manager that checkpoints WAL on close."""
    conn = init_db(db_path)
    try:
        yield conn
    finally:
        conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        conn.close()
