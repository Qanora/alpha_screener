"""Tests for SQLite database initialization."""

import sqlite3
import tempfile
from pathlib import Path

from alphascreener.db import get_db, init_db


class TestInitDb:
    def test_creates_directory_and_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "sub" / "test.db"
            conn = init_db(db_path)
            assert db_path.exists()
            conn.close()

    def test_enables_wal_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            conn = init_db(db_path)
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode.lower() == "wal"
            conn.close()

    def test_creates_all_tables(self):
        expected_tables = {
            "factor_versions",
            "paper_trades",
            "alerts",
            "llm_cost_daily",
            "pid_lock",
            "monitoring_samples",
            "alpha_acceptance_daily",
            "data_source_diff",
            "factor_health_daily",
        }
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            conn = init_db(db_path)
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            table_names = {t[0] for t in tables} - {"sqlite_sequence"}
            assert expected_tables == table_names
            conn.close()

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            conn1 = init_db(db_path)
            conn1.close()
            conn2 = init_db(db_path)
            conn2.close()
            assert db_path.exists()


class TestGetDb:
    def test_context_manager_yields_connection(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            with get_db(db_path) as conn:
                assert isinstance(conn, sqlite3.Connection)
                conn.execute("SELECT 1")

    def test_data_persists_after_context_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            with get_db(db_path) as conn:
                conn.execute(
                    "INSERT INTO alerts (triggered_at, severity, rule_name) VALUES (datetime('now'), 'warning', 'test')"
                )
                conn.commit()
            # After context exit, data is readable from a new connection
            conn2 = sqlite3.connect(str(db_path))
            conn2.execute("PRAGMA journal_mode=WAL")
            count = conn2.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
            assert count == 1
            conn2.close()


class TestDDL:
    def test_paper_trades_foreign_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            conn = init_db(db_path)
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(
                "INSERT INTO factor_versions (version, released_at, config_json, release_type) VALUES ('1.0.0', datetime('now'), '{}', 'MAJOR')"
            )
            conn.execute(
                "INSERT INTO paper_trades (signal_date, ticker, rating, breakout_probability, factor_version) VALUES (date('now'), 'AAPL', 'Buy', 0.65, '1.0.0')"
            )
            conn.commit()
            row = conn.execute("SELECT ticker FROM paper_trades WHERE ticker='AAPL'").fetchone()
            assert row[0] == "AAPL"
            conn.close()

    def test_alerts_severity_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            conn = init_db(db_path)
            conn.execute(
                "INSERT INTO alerts (triggered_at, severity, rule_name) VALUES (datetime('now'), 'critical', 'test_rule')"
            )
            conn.commit()
            row = conn.execute("SELECT severity FROM alerts").fetchone()
            assert row[0] == "critical"
            conn.close()
