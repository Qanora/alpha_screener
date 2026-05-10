"""Tests for CLI commands."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from alphascreener.cli import main


class TestCliBasics:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "AlphaScreener" in result.output

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0

    def test_init_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("alphascreener.cli.get_settings") as mock_settings:
                mock_settings.return_value.home = Path(tmp)
                mock_settings.return_value.db_path = Path(tmp) / "db" / "metadata.db"
                runner = CliRunner()
                result = runner.invoke(main, ["init-db"])
                assert result.exit_code == 0
                assert "Database initialized" in result.output

    def test_screen_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["screen", "--help"])
        assert result.exit_code == 0
        assert "--market" in result.output
        assert "--top" in result.output

    def test_backtest_requires_start(self):
        runner = CliRunner()
        result = runner.invoke(main, ["backtest"])
        assert result.exit_code != 0

    def test_backtest_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["backtest", "--help"])
        assert result.exit_code == 0
        assert "--start" in result.output


class TestDataCommands:
    def test_data_group_exists(self):
        runner = CliRunner()
        result = runner.invoke(main, ["data", "--help"])
        assert result.exit_code == 0
        assert "sync" in result.output
        assert "universe" in result.output

    def test_data_universe_no_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("alphascreener.cli.get_settings") as mock_settings:
                mock_settings.return_value.home = Path(tmp)
                runner = CliRunner()
                result = runner.invoke(main, ["data", "universe"])
                assert result.exit_code == 0
                assert "No universe data" in result.output

    def test_sync_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["data", "sync", "--help"])
        assert result.exit_code == 0
        assert "--days" in result.output
        assert "--top" in result.output
