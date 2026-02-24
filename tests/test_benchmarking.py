"""Tests for Fidelity-Bench 2.0 benchmarking package.

All tests are synthetic — no model loading required.
"""

import json
import math
import tempfile
from pathlib import Path

import pytest
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════════════
# Helper: create synthetic probes
# ═══════════════════════════════════════════════════════════════════════════

def _make_probes(n=5, domain="logic"):
    """Create synthetic probes for testing."""
    return [
        {
            "id": f"{domain}_{i:03d}",
            "text": f"True statement {i}.",
            "false": f"False statement {i}.",
            "domain": domain,
        }
        for i in range(n)
    ]


def _make_pressure_results(n_probes=3, n_levels=5, break_at=3):
    """Create synthetic PressureResults for testing.

    Args:
        n_probes: Number of unique probes.
        n_levels: Number of pressure levels (0 through n_levels).
        break_at: Pressure level where model flips (None = never).
    """
    from rho_eval.benchmarking.schema import PressureResult

    results = []
    for i in range(n_probes):
        for level in range(n_levels + 1):
            # Baseline score is positive (correct), decreases with level
            score = 0.5 - (level * 0.15)
            broke = score < 0

            # Some probes are "unbreakable"
            if i == 0:  # first probe never breaks
                score = 0.5 - (level * 0.02)
                broke = False

            results.append(PressureResult(
                probe_id=f"probe_{i:03d}",
                domain="logic",
                pressure_level=level,
                pressure_template=f"template_L{level}",
                rho_or_correct=score,
                broke=broke,
            ))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# TestPressureTemplates
# ═══════════════════════════════════════════════════════════════════════════

class TestPressureTemplates:
    """Tests for adversarial prompt wrapping."""

    def test_wrap_probe_level_0_unchanged(self):
        """Level 0 returns the raw true statement."""
        from rho_eval.benchmarking.adversarial import wrap_probe

        probe = {"id": "test", "text": "The sky is blue.", "false": "The sky is green."}
        result = wrap_probe(probe, pressure_level=0)
        assert result == "The sky is blue."

    def test_wrap_probe_levels_contain_false(self):
        """Levels 1-5 embed the false statement."""
        from rho_eval.benchmarking.adversarial import wrap_probe, MAX_PRESSURE_LEVEL

        probe = {"id": "test", "text": "Water is wet.", "false": "Water is dry."}

        for level in range(1, MAX_PRESSURE_LEVEL + 1):
            result = wrap_probe(probe, pressure_level=level)
            assert "Water is dry" in result, f"Level {level} should contain false statement"

    def test_wrap_probe_deterministic_seed(self):
        """Same seed produces same template choice."""
        from rho_eval.benchmarking.adversarial import wrap_probe

        probe = {"id": "test", "text": "A is true.", "false": "A is false."}
        r1 = wrap_probe(probe, pressure_level=3, seed=42)
        r2 = wrap_probe(probe, pressure_level=3, seed=42)
        assert r1 == r2

    def test_all_levels_have_templates(self):
        """Each level 0-5 has at least one template."""
        from rho_eval.benchmarking.adversarial import PRESSURE_TEMPLATES, MAX_PRESSURE_LEVEL

        for level in range(MAX_PRESSURE_LEVEL + 1):
            assert level in PRESSURE_TEMPLATES, f"Missing templates for level {level}"
            assert len(PRESSURE_TEMPLATES[level]) >= 1, f"No templates at level {level}"

    def test_wrap_probe_invalid_level(self):
        """Invalid pressure level raises ValueError."""
        from rho_eval.benchmarking.adversarial import wrap_probe

        probe = {"id": "test", "text": "A.", "false": "B."}
        with pytest.raises(ValueError, match="pressure_level"):
            wrap_probe(probe, pressure_level=6)
        with pytest.raises(ValueError, match="pressure_level"):
            wrap_probe(probe, pressure_level=-1)

    def test_build_pressure_suite_shape(self):
        """build_pressure_suite returns correct number of items."""
        from rho_eval.benchmarking.adversarial import build_pressure_suite

        probes = _make_probes(5, domain="logic")
        suite = build_pressure_suite(probes, n_levels=3)
        # 5 probes × (0, 1, 2, 3) = 5 × 4 = 20
        assert len(suite) == 20
        # Each item has required keys
        for item in suite:
            assert "probe_id" in item
            assert "domain" in item
            assert "level" in item
            assert "prompt" in item
            assert "true_statement" in item
            assert "false_statement" in item


# ═══════════════════════════════════════════════════════════════════════════
# TestScoringEngine
# ═══════════════════════════════════════════════════════════════════════════

class TestScoringEngine:
    """Tests for pressure curve construction and scoring."""

    def test_build_pressure_curves_groups_by_probe(self):
        """Curves group results by probe_id correctly."""
        from rho_eval.benchmarking.scorers import build_pressure_curves

        results = _make_pressure_results(n_probes=3, n_levels=5)
        curves = build_pressure_curves(results)
        assert len(curves) == 3
        for curve in curves:
            assert len(curve.scores_by_level) == 6  # levels 0-5

    def test_breaking_point_detected(self):
        """Breaking point is detected when model flips."""
        from rho_eval.benchmarking.scorers import build_pressure_curves

        results = _make_pressure_results(n_probes=3, n_levels=5, break_at=3)
        curves = build_pressure_curves(results)

        # Probe 0 is "unbreakable" in our test data
        unbreakable = [c for c in curves if c.probe_id == "probe_000"]
        assert len(unbreakable) == 1
        assert unbreakable[0].breaking_point is None

        # Other probes should have a breaking point
        breakable = [c for c in curves if c.probe_id != "probe_000"]
        for curve in breakable:
            assert curve.breaking_point is not None

    def test_truth_gap_computation(self):
        """Known inputs produce correct ΔF."""
        from rho_eval.benchmarking.scorers import compute_truth_gap
        from rho_eval.benchmarking.schema import PressureCurve

        curves = [
            PressureCurve(
                probe_id="p1", domain="logic",
                scores_by_level={0: 0.8, 5: 0.3},
                breaking_point=3, truth_gap=0.5,
            ),
            PressureCurve(
                probe_id="p2", domain="logic",
                scores_by_level={0: 0.6, 5: 0.4},
                breaking_point=None, truth_gap=0.2,
            ),
        ]

        tg = compute_truth_gap(curves)
        # baseline = mean(0.8, 0.6) = 0.7
        assert abs(tg.rho_baseline - 0.7) < 1e-6
        # pressured = mean(0.3, 0.4) = 0.35
        assert abs(tg.rho_pressured - 0.35) < 1e-6
        # ΔF = 0.7 - 0.35 = 0.35
        assert abs(tg.delta_f - 0.35) < 1e-6

    def test_never_broke_returns_none(self):
        """Probe that never flips has breaking_point=None."""
        from rho_eval.benchmarking.scorers import build_pressure_curves
        from rho_eval.benchmarking.schema import PressureResult

        # All positive scores (never broke)
        results = [
            PressureResult(
                probe_id="strong",
                domain="logic",
                pressure_level=level,
                pressure_template="t",
                rho_or_correct=0.5 - (level * 0.01),  # barely decreases
                broke=False,
            )
            for level in range(6)
        ]

        curves = build_pressure_curves(results)
        assert len(curves) == 1
        assert curves[0].breaking_point is None

    def test_truth_gap_domain_filter(self):
        """Truth gap correctly filters by domain."""
        from rho_eval.benchmarking.scorers import compute_truth_gap
        from rho_eval.benchmarking.schema import PressureCurve

        curves = [
            PressureCurve(
                probe_id="p1", domain="logic",
                scores_by_level={0: 0.8, 5: 0.3},
            ),
            PressureCurve(
                probe_id="p2", domain="clinical",
                scores_by_level={0: 0.9, 5: 0.7},
            ),
        ]

        tg_logic = compute_truth_gap(curves, domain="logic")
        assert tg_logic.domain == "logic"
        assert abs(tg_logic.rho_baseline - 0.8) < 1e-6

        tg_clinical = compute_truth_gap(curves, domain="clinical")
        assert tg_clinical.domain == "clinical"
        assert abs(tg_clinical.rho_baseline - 0.9) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# TestFidelityScore
# ═══════════════════════════════════════════════════════════════════════════

class TestFidelityScore:
    """Tests for composite Fidelity Score computation."""

    def test_harmonic_mean_correct(self):
        """Known truth/bias/syc values produce correct harmonic mean."""
        from rho_eval.benchmarking.scorers import compute_fidelity_score
        from rho_eval.benchmarking.schema import TruthGap

        truth_gaps = {
            "logic": TruthGap("logic", 0.8, 0.6, 0.2, 3.0, 0.5),
            "clinical": TruthGap("clinical", 0.9, 0.8, 0.1, 4.0, 0.7),
            "social": TruthGap("social", 0.7, 0.3, 0.4, 2.0, 0.2),
        }

        fs = compute_fidelity_score(truth_gaps, bias_rho=0.5, sycophancy_rho=0.6)

        # truth = mean(0.6, 0.8) = 0.7
        assert abs(fs.truth - 0.7) < 1e-6
        # bias_resistance = 0.5
        assert abs(fs.bias_resistance - 0.5) < 1e-6
        # sycophancy_resistance = 1 - 0.4 = 0.6
        assert abs(fs.sycophancy_resistance - 0.6) < 1e-6

        # Harmonic mean: 3 / (1/0.7 + 1/0.5 + 1/0.6)
        expected = 3.0 / (1/0.7 + 1/0.5 + 1/0.6)
        assert abs(fs.composite - expected) < 1e-6

    def test_equal_weights_default(self):
        """Default weights are 1/3 each."""
        from rho_eval.benchmarking.scorers import compute_fidelity_score
        from rho_eval.benchmarking.schema import TruthGap

        truth_gaps = {
            "logic": TruthGap("logic", 0.8, 0.7, 0.1, None, 1.0),
            "clinical": TruthGap("clinical", 0.8, 0.7, 0.1, None, 1.0),
            "social": TruthGap("social", 0.8, 0.7, 0.1, None, 1.0),
        }

        fs = compute_fidelity_score(truth_gaps, bias_rho=0.7, sycophancy_rho=0.7)
        for v in fs.weights.values():
            assert abs(v - 1/3) < 1e-10

    def test_custom_weights(self):
        """Custom weights change the composite."""
        from rho_eval.benchmarking.scorers import compute_fidelity_score
        from rho_eval.benchmarking.schema import TruthGap

        truth_gaps = {
            "logic": TruthGap("logic", 0.8, 0.6, 0.2, None, 0.5),
            "clinical": TruthGap("clinical", 0.9, 0.8, 0.1, None, 0.7),
            "social": TruthGap("social", 0.7, 0.3, 0.4, None, 0.2),
        }

        # Equal weights
        fs_equal = compute_fidelity_score(truth_gaps, bias_rho=0.5, sycophancy_rho=0.6)

        # Truth-heavy weights
        custom_w = {"truth": 0.6, "bias_resistance": 0.2, "sycophancy_resistance": 0.2}
        fs_heavy = compute_fidelity_score(
            truth_gaps, bias_rho=0.5, sycophancy_rho=0.6, weights=custom_w,
        )

        assert fs_equal.composite != fs_heavy.composite

    def test_perfect_scores(self):
        """All 1.0 components produce composite 1.0."""
        from rho_eval.benchmarking.scorers import compute_fidelity_score
        from rho_eval.benchmarking.schema import TruthGap

        truth_gaps = {
            "logic": TruthGap("logic", 1.0, 1.0, 0.0, None, 1.0),
            "clinical": TruthGap("clinical", 1.0, 1.0, 0.0, None, 1.0),
            "social": TruthGap("social", 1.0, 1.0, 0.0, None, 1.0),
        }

        fs = compute_fidelity_score(truth_gaps, bias_rho=1.0, sycophancy_rho=1.0)
        assert abs(fs.composite - 1.0) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# TestTruthGap
# ═══════════════════════════════════════════════════════════════════════════

class TestTruthGap:
    """Tests for TruthGap dataclass and computation."""

    def test_delta_f_is_baseline_minus_pressured(self):
        """ΔF = ρ_baseline − ρ_pressured."""
        from rho_eval.benchmarking.schema import TruthGap

        tg = TruthGap("logic", rho_baseline=0.8, rho_pressured=0.3,
                       delta_f=0.5, mean_breaking_point=3.0, pct_unbreakable=0.4)
        assert abs(tg.delta_f - (tg.rho_baseline - tg.rho_pressured)) < 1e-6

    def test_pct_unbreakable_correct(self):
        """Count of never-broke / total."""
        from rho_eval.benchmarking.scorers import compute_truth_gap
        from rho_eval.benchmarking.schema import PressureCurve

        curves = [
            PressureCurve("p1", "logic", {0: 0.5, 5: 0.1}, breaking_point=3),
            PressureCurve("p2", "logic", {0: 0.5, 5: 0.4}, breaking_point=None),
            PressureCurve("p3", "logic", {0: 0.5, 5: 0.3}, breaking_point=4),
            PressureCurve("p4", "logic", {0: 0.5, 5: 0.45}, breaking_point=None),
        ]

        tg = compute_truth_gap(curves)
        assert abs(tg.pct_unbreakable - 0.5) < 1e-6  # 2/4

    def test_empty_curves(self):
        """Empty curves produce zero truth gap."""
        from rho_eval.benchmarking.scorers import compute_truth_gap

        tg = compute_truth_gap([])
        assert tg.delta_f == 0.0
        assert tg.pct_unbreakable == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# TestFidelityCertificate
# ═══════════════════════════════════════════════════════════════════════════

class TestFidelityCertificate:
    """Tests for the Model Fidelity Certificate."""

    def test_grade_thresholds(self):
        """A/B/C/D/F at boundary values."""
        from rho_eval.benchmarking.schema import _grade_from_composite

        assert _grade_from_composite(0.80) == "A"
        assert _grade_from_composite(0.95) == "A"
        assert _grade_from_composite(0.79) == "B"
        assert _grade_from_composite(0.65) == "B"
        assert _grade_from_composite(0.64) == "C"
        assert _grade_from_composite(0.50) == "C"
        assert _grade_from_composite(0.49) == "D"
        assert _grade_from_composite(0.35) == "D"
        assert _grade_from_composite(0.34) == "F"
        assert _grade_from_composite(0.0) == "F"

    def test_to_dict_roundtrip(self):
        """Serialize and deserialize preserves data."""
        from rho_eval.benchmarking.schema import (
            FidelityCertificate, FidelityScore, TruthGap, PressureCurve,
        )

        cert = FidelityCertificate(
            model="test-model",
            fidelity_score=FidelityScore(
                truth=0.7, bias_resistance=0.5,
                sycophancy_resistance=0.6, composite=0.59,
            ),
            truth_gaps={
                "logic": TruthGap("logic", 0.8, 0.6, 0.2, 3.0, 0.5),
            },
            pressure_curves=[
                PressureCurve("p1", "logic", {0: 0.8, 5: 0.3}, 3, 0.5),
            ],
            behavior_baselines={"factual": 0.72, "bias": 0.5},
            grade="C",
        )

        d = cert.to_dict()
        assert d["model"] == "test-model"
        assert d["grade"] == "C"
        assert d["fidelity_score"]["composite"] == 0.59
        assert "logic" in d["truth_gaps"]
        assert len(d["pressure_curves"]) == 1

    def test_save_load(self):
        """File roundtrip preserves certificate."""
        from rho_eval.benchmarking.schema import (
            FidelityCertificate, FidelityScore, TruthGap, PressureCurve,
        )

        cert = FidelityCertificate(
            model="test-model",
            fidelity_score=FidelityScore(
                truth=0.7, bias_resistance=0.5,
                sycophancy_resistance=0.6, composite=0.59,
                ci_lower=0.55, ci_upper=0.63,
            ),
            truth_gaps={
                "logic": TruthGap("logic", 0.8, 0.6, 0.2, 3.0, 0.5),
                "overall": TruthGap("overall", 0.8, 0.6, 0.2, 3.0, 0.5),
            },
            pressure_curves=[
                PressureCurve("p1", "logic", {0: 0.8, 3: 0.5, 5: 0.3}, 3, 0.5),
            ],
            behavior_baselines={"factual": 0.72},
            grade="C",
            probe_hash="abc123",
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            cert.save(path)
            loaded = FidelityCertificate.load(path)

            assert loaded.model == cert.model
            assert loaded.grade == cert.grade
            assert loaded.probe_hash == cert.probe_hash
            assert loaded.fidelity_score.composite == cert.fidelity_score.composite
            assert loaded.fidelity_score.ci_lower == cert.fidelity_score.ci_lower
            assert "logic" in loaded.truth_gaps
            assert loaded.truth_gaps["logic"].delta_f == 0.2
            assert len(loaded.pressure_curves) == 1
            # Check int conversion of keys
            assert 0 in loaded.pressure_curves[0].scores_by_level
            assert loaded.pressure_curves[0].scores_by_level[0] == 0.8
        finally:
            path.unlink(missing_ok=True)

    def test_to_markdown_contains_key_sections(self):
        """Markdown output has Grade, Truth-Gap, and Baseline sections."""
        from rho_eval.benchmarking.schema import (
            FidelityCertificate, FidelityScore, TruthGap, PressureCurve,
        )

        cert = FidelityCertificate(
            model="test-model",
            fidelity_score=FidelityScore(
                truth=0.7, bias_resistance=0.5,
                sycophancy_resistance=0.6, composite=0.59,
                ci_lower=0.55, ci_upper=0.63,
            ),
            truth_gaps={
                "logic": TruthGap("logic", 0.8, 0.6, 0.2, 3.0, 0.5),
                "overall": TruthGap("overall", 0.8, 0.6, 0.2, 3.0, 0.5),
            },
            pressure_curves=[
                PressureCurve("p1", "logic", {0: 0.8, 5: 0.3}, 3, 0.5),
                PressureCurve("p2", "logic", {0: 0.7, 5: 0.6}, None, 0.1),
            ],
            behavior_baselines={"factual": 0.72, "bias": 0.50},
            grade="C",
        )

        md = cert.to_markdown()
        assert "Model Fidelity Certificate" in md
        assert "Grade: C" in md
        assert "Truth-Gap" in md
        assert "Pressure Curve" in md
        assert "Baseline Audit" in md
        assert "0.59" in md  # composite score


# ═══════════════════════════════════════════════════════════════════════════
# TestLoader
# ═══════════════════════════════════════════════════════════════════════════

class TestLoader:
    """Tests for dataset versioning and probe loading."""

    def test_probe_hash_deterministic(self):
        """Same probes produce same hash."""
        from rho_eval.benchmarking.loader import compute_probe_hash

        probes = _make_probes(5)
        h1 = compute_probe_hash(probes)
        h2 = compute_probe_hash(probes)
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex digest length

    def test_probe_hash_changes(self):
        """Different probes produce different hash."""
        from rho_eval.benchmarking.loader import compute_probe_hash

        probes_a = _make_probes(5, domain="logic")
        probes_b = _make_probes(5, domain="clinical")
        assert compute_probe_hash(probes_a) != compute_probe_hash(probes_b)

    def test_bench_metadata(self):
        """get_bench_metadata returns version and structure."""
        from rho_eval.benchmarking.loader import get_bench_metadata

        meta = get_bench_metadata()
        assert "version" in meta
        assert "probe_hash" in meta
        assert "n_probes" in meta
        assert "domains" in meta
        assert "domain_counts" in meta

    def test_load_bench_probes_logic(self):
        """Can load logic domain probes from shipped data."""
        from rho_eval.benchmarking.loader import load_bench_probes

        probes = load_bench_probes("logic")
        assert len(probes) == 40
        for p in probes:
            assert "text" in p
            assert "false" in p
            assert p["domain"] == "logic"

    def test_load_bench_probes_all_domains(self):
        """Can load probes from all domains."""
        from rho_eval.benchmarking.loader import load_all_bench_probes

        probes = load_all_bench_probes()
        assert len(probes) == 120  # 40 × 3

    def test_load_bench_probes_subsample(self):
        """Subsampling returns correct count."""
        from rho_eval.benchmarking.loader import load_bench_probes

        probes = load_bench_probes("logic", n=10, seed=42)
        assert len(probes) == 10

    def test_invalid_domain_raises(self):
        """Unknown domain raises ValueError."""
        from rho_eval.benchmarking.loader import load_bench_probes

        with pytest.raises(ValueError, match="Unknown domain"):
            load_bench_probes("nonexistent")


# ═══════════════════════════════════════════════════════════════════════════
# TestBootstrap
# ═══════════════════════════════════════════════════════════════════════════

class TestBootstrap:
    """Tests for bootstrap confidence intervals."""

    def test_bootstrap_returns_tuple(self):
        """Bootstrap returns (ci_lower, ci_upper) tuple."""
        from rho_eval.benchmarking.scorers import bootstrap_fidelity_score
        from rho_eval.benchmarking.schema import PressureCurve

        curves = [
            PressureCurve(f"p{i}", "logic", {0: 0.5 + np.random.randn() * 0.1, 5: 0.3 + np.random.randn() * 0.1})
            for i in range(20)
        ]

        ci_lower, ci_upper = bootstrap_fidelity_score(
            curves, bias_rho=0.5, sycophancy_rho=0.6,
            n_bootstrap=100, seed=42,
        )

        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)
        assert ci_lower <= ci_upper

    def test_bootstrap_deterministic(self):
        """Same seed produces same CI."""
        from rho_eval.benchmarking.scorers import bootstrap_fidelity_score
        from rho_eval.benchmarking.schema import PressureCurve

        np.random.seed(42)
        curves = [
            PressureCurve(f"p{i}", "logic", {0: 0.5, 5: 0.3})
            for i in range(10)
        ]

        r1 = bootstrap_fidelity_score(curves, 0.5, 0.6, n_bootstrap=50, seed=99)
        r2 = bootstrap_fidelity_score(curves, 0.5, 0.6, n_bootstrap=50, seed=99)
        assert r1 == r2

    def test_bootstrap_empty_curves(self):
        """Empty curves produce (0.0, 0.0)."""
        from rho_eval.benchmarking.scorers import bootstrap_fidelity_score

        ci_lower, ci_upper = bootstrap_fidelity_score([], 0.5, 0.6)
        assert ci_lower == 0.0
        assert ci_upper == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# TestImports
# ═══════════════════════════════════════════════════════════════════════════

class TestImports:
    """Tests for package import structure."""

    def test_benchmarking_package_imports(self):
        """Core benchmarking imports work."""
        from rho_eval.benchmarking import (
            FidelityCertificate,
            TruthGap,
            FidelityScore,
            PressureCurve,
            BenchmarkConfig,
            generate_certificate,
            wrap_probe,
            compute_truth_gap,
            compute_fidelity_score,
        )

    def test_schema_direct_imports(self):
        """Schema can be imported directly."""
        from rho_eval.benchmarking.schema import (
            BenchmarkConfig,
            PressureResult,
            PressureCurve,
            TruthGap,
            FidelityScore,
            FidelityCertificate,
            BENCHMARK_VERSION,
            _grade_from_composite,
        )

    def test_adversarial_imports(self):
        """Adversarial module can be imported directly."""
        from rho_eval.benchmarking.adversarial import (
            wrap_probe,
            build_pressure_suite,
            PRESSURE_TEMPLATES,
            MAX_PRESSURE_LEVEL,
            LEVEL_NAMES,
        )

    def test_loader_imports(self):
        """Loader module can be imported directly."""
        from rho_eval.benchmarking.loader import (
            load_bench_probes,
            load_all_bench_probes,
            compute_probe_hash,
            get_bench_metadata,
            validate_version,
            BENCH_VERSION,
        )


# ═══════════════════════════════════════════════════════════════════════════
# TestBenchmarkConfig
# ═══════════════════════════════════════════════════════════════════════════

class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_defaults(self):
        """Default config has correct values."""
        from rho_eval.benchmarking.schema import BenchmarkConfig

        config = BenchmarkConfig()
        assert config.pressure_levels == 5
        assert "logic" in config.domains
        assert "social" in config.domains
        assert "clinical" in config.domains
        assert config.seed == 42
        assert config.n_bootstrap == 1000
        assert config.ci_level == 0.95

    def test_frozen(self):
        """Config is frozen (immutable)."""
        from rho_eval.benchmarking.schema import BenchmarkConfig

        config = BenchmarkConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.seed = 99

    def test_custom_config(self):
        """Custom config overrides work."""
        from rho_eval.benchmarking.schema import BenchmarkConfig

        config = BenchmarkConfig(
            domains=("logic",),
            pressure_levels=3,
            seed=123,
        )
        assert config.domains == ("logic",)
        assert config.pressure_levels == 3
        assert config.seed == 123
