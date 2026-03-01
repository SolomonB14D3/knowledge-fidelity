"""Tests for the CLI interface."""

import subprocess
import sys
import pytest


class TestCLIInfoCommands:
    """Test CLI commands that don't require a model."""

    def test_version(self):
        from rho_eval import __version__
        result = subprocess.run(
            [sys.executable, "-m", "rho_eval.cli.rho_audit", "--version"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "rho-eval" in result.stdout
        assert __version__ in result.stdout

    def test_list_behaviors(self):
        result = subprocess.run(
            [sys.executable, "-m", "rho_eval.cli.rho_audit", "--list-behaviors"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "factual" in result.stdout
        assert "toxicity" in result.stdout
        assert "bias" in result.stdout
        assert "sycophancy" in result.stdout
        assert "reasoning" in result.stdout

    def test_list_probes(self):
        result = subprocess.run(
            [sys.executable, "-m", "rho_eval.cli.rho_audit", "--list-probes"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "factual/default" in result.stdout
        assert "TOTAL" in result.stdout

    def test_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "rho_eval.cli.rho_audit", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "behavioral audit" in result.stdout.lower()
        assert "--behaviors" in result.stdout
        assert "--format" in result.stdout

    def test_no_model_shows_error(self):
        result = subprocess.run(
            [sys.executable, "-m", "rho_eval.cli.rho_audit"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0


class TestCLIArgumentParsing:
    """Test that CLI arguments are parsed correctly (no model needed)."""

    def test_format_options(self):
        """Verify format choices are accepted (via --help)."""
        result = subprocess.run(
            [sys.executable, "-m", "rho_eval.cli.rho_audit", "--help"],
            capture_output=True, text=True,
        )
        assert "table" in result.stdout
        assert "json" in result.stdout
        assert "markdown" in result.stdout
        assert "csv" in result.stdout
