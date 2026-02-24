"""Tests for the output module — AuditReport, exporters, comparator."""

import json
import tempfile
from pathlib import Path

import pytest


class TestBehaviorResult:
    """Test the BehaviorResult dataclass."""

    def test_status_pass(self):
        from rho_eval.behaviors.base import BehaviorResult
        r = BehaviorResult("test", rho=0.65, retention=0.8, positive_count=40, total=50)
        assert r.status == "PASS"

    def test_status_warn(self):
        from rho_eval.behaviors.base import BehaviorResult
        r = BehaviorResult("test", rho=0.35, retention=0.3, positive_count=15, total=50)
        assert r.status == "WARN"

    def test_status_fail(self):
        from rho_eval.behaviors.base import BehaviorResult
        r = BehaviorResult("test", rho=0.1, retention=0.1, positive_count=5, total=50)
        assert r.status == "FAIL"

    def test_to_dict(self):
        from rho_eval.behaviors.base import BehaviorResult
        r = BehaviorResult("test", rho=0.5, retention=0.5, positive_count=25, total=50)
        d = r.to_dict()
        assert d["behavior"] == "test"
        assert d["rho"] == 0.5
        assert d["total"] == 50

    def test_summary_line(self):
        from rho_eval.behaviors.base import BehaviorResult
        r = BehaviorResult("test", rho=0.5, retention=0.5, positive_count=25, total=50, elapsed=1.0)
        line = r.summary_line()
        assert "test" in line
        assert "PASS" in line
        assert "1.0s" in line


class TestAuditReport:
    """Test the AuditReport dataclass."""

    def test_create_empty(self):
        from rho_eval.output.schema import AuditReport
        r = AuditReport(model="test")
        assert r.model == "test"
        assert len(r.behaviors) == 0
        assert r.mean_rho == 0.0
        assert r.overall_status == "FAIL"

    def test_add_result(self, mock_audit_report):
        assert len(mock_audit_report.behaviors) == 3
        assert "factual" in mock_audit_report.behaviors
        assert "toxicity" in mock_audit_report.behaviors
        assert "bias" in mock_audit_report.behaviors

    def test_mean_rho(self, mock_audit_report):
        expected = (0.72 + 0.61 + 0.15) / 3
        assert abs(mock_audit_report.mean_rho - expected) < 1e-10

    def test_overall_status_fail_if_any_fail(self, mock_audit_report):
        # bias has rho=0.15 → FAIL
        assert mock_audit_report.overall_status == "FAIL"

    def test_overall_status_pass(self):
        from rho_eval.behaviors.base import BehaviorResult
        from rho_eval.output.schema import AuditReport

        r = AuditReport(model="good-model")
        r.add_result(BehaviorResult("a", rho=0.8, retention=0.8, positive_count=40, total=50))
        r.add_result(BehaviorResult("b", rho=0.7, retention=0.7, positive_count=35, total=50))
        assert r.overall_status == "PASS"

    def test_total_probes(self, mock_audit_report):
        assert mock_audit_report.total_probes == 56 + 100 + 300

    def test_to_dict(self, mock_audit_report):
        d = mock_audit_report.to_dict()
        assert d["model"] == "test-model"
        assert "summary" in d
        assert "behaviors" in d
        assert d["summary"]["n_behaviors"] == 3

    def test_to_dict_without_details(self, mock_audit_report):
        d = mock_audit_report.to_dict(include_details=False)
        for name, bd in d["behaviors"].items():
            assert "details" not in bd

    def test_to_json(self, mock_audit_report):
        j = mock_audit_report.to_json()
        d = json.loads(j)
        assert d["model"] == "test-model"

    def test_save_and_load_roundtrip(self, mock_audit_report):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        mock_audit_report.save(path)
        loaded = type(mock_audit_report).load(path)

        assert loaded.model == mock_audit_report.model
        assert len(loaded.behaviors) == len(mock_audit_report.behaviors)
        assert abs(loaded.mean_rho - mock_audit_report.mean_rho) < 1e-10

        for name in mock_audit_report.behaviors:
            assert name in loaded.behaviors
            assert abs(loaded.behaviors[name].rho - mock_audit_report.behaviors[name].rho) < 1e-10

        Path(path).unlink()

    def test_repr(self, mock_audit_report):
        r = repr(mock_audit_report)
        assert "test-model" in r
        assert "behaviors=3" in r


class TestExporters:
    """Test export formats."""

    def test_to_markdown(self, mock_audit_report):
        from rho_eval.output.exporters import to_markdown
        md = to_markdown(mock_audit_report)
        assert "# Behavioral Audit:" in md
        assert "factual" in md
        assert "|" in md

    def test_to_csv(self, mock_audit_report):
        from rho_eval.output.exporters import to_csv
        csv_str = to_csv(mock_audit_report)
        lines = csv_str.strip().split("\n")
        assert len(lines) == 4  # header + 3 behaviors
        assert "model" in lines[0]
        assert "rho" in lines[0]

    def test_to_table(self, mock_audit_report):
        from rho_eval.output.exporters import to_table
        table = to_table(mock_audit_report, color=False)
        assert "Behavioral Audit:" in table
        assert "factual" in table
        assert "PASS" in table
        assert "FAIL" in table


class TestComparator:
    """Test the comparison system."""

    def test_compare_basic(self, mock_audit_report):
        from rho_eval.behaviors.base import BehaviorResult
        from rho_eval.output.schema import AuditReport
        from rho_eval.output.comparator import compare

        report2 = AuditReport(model="improved-model")
        report2.add_result(BehaviorResult("factual", rho=0.80, retention=0.90, positive_count=50, total=56))
        report2.add_result(BehaviorResult("toxicity", rho=0.55, retention=0.60, positive_count=60, total=100))
        report2.add_result(BehaviorResult("bias", rho=0.20, retention=0.20, positive_count=60, total=300))

        result = compare(report2, mock_audit_report)
        assert len(result.deltas) == 3

        # factual improved
        factual_d = next(d for d in result.deltas if d.behavior == "factual")
        assert factual_d.label == "IMPROVED"
        assert factual_d.delta_rho > 0

        # toxicity degraded
        tox_d = next(d for d in result.deltas if d.behavior == "toxicity")
        assert tox_d.label == "DEGRADED"
        assert tox_d.delta_rho < 0

    def test_compare_to_table(self, mock_audit_report):
        from rho_eval.behaviors.base import BehaviorResult
        from rho_eval.output.schema import AuditReport
        from rho_eval.output.comparator import compare

        report2 = AuditReport(model="model2")
        report2.add_result(BehaviorResult("factual", rho=0.72, retention=0.85, positive_count=48, total=56))
        report2.add_result(BehaviorResult("toxicity", rho=0.61, retention=0.70, positive_count=70, total=100))
        report2.add_result(BehaviorResult("bias", rho=0.15, retention=0.15, positive_count=45, total=300))

        result = compare(report2, mock_audit_report)
        table = result.to_table(color=False)
        assert "Comparison:" in table

        md = result.to_markdown()
        assert "## Comparison:" in md
