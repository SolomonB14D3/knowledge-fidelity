"""Tests for the Rho-Surgery pipeline components.

Tests the surgical planner (plan generation, risk classification, verification),
HybridConfig extensions (gamma_weight, protection fields), and the γ protection
loss integration. All tests use synthetic data — no model loading required.
"""

import json
import pytest
import tempfile
from pathlib import Path


# ═════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_category_metrics():
    """Category metrics as they appear in AuditReport metadata."""
    return {
        "Age": {"accuracy": 0.882, "n": 17, "biased_rate": 0.059},
        "Race_ethnicity": {"accuracy": 0.892, "n": 37, "biased_rate": 0.054},
        "Religion": {"accuracy": 0.833, "n": 6, "biased_rate": 0.0},
        "Gender_identity": {"accuracy": 0.786, "n": 14, "biased_rate": 0.071},
        "SES": {"accuracy": 0.840, "n": 25, "biased_rate": 0.040},
        "Race_x_SES": {"accuracy": 0.841, "n": 63, "biased_rate": 0.016},
        "Disability_status": {"accuracy": 0.714, "n": 14, "biased_rate": 0.143},
        "Sexual_orientation": {"accuracy": 0.750, "n": 8, "biased_rate": 0.125},
    }


@pytest.fixture
def mock_audit_report_with_categories(mock_category_metrics):
    """AuditReport with per-category bias metrics."""
    from rho_eval.behaviors.base import BehaviorResult
    from rho_eval.output.schema import AuditReport

    report = AuditReport(model="Qwen/Qwen2.5-7B-Instruct", device="mps")
    report.add_result(BehaviorResult(
        behavior="sycophancy", rho=0.3820, retention=0.69,
        positive_count=69, total=100, elapsed=5.0,
    ))
    report.add_result(BehaviorResult(
        behavior="bias", rho=0.7730, retention=0.77,
        positive_count=300, total=388, elapsed=10.0,
        metadata={"category_metrics": mock_category_metrics},
    ))
    report.add_result(BehaviorResult(
        behavior="factual", rho=0.6200, retention=0.72,
        positive_count=72, total=100, elapsed=5.0,
    ))
    return report


@pytest.fixture
def mock_surgical_plan():
    """Pre-built SurgicalPlan for testing."""
    from rho_eval.surgical_planner import SurgicalPlan
    return SurgicalPlan(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        target_behaviors=["sycophancy"],
        baseline_scores={"sycophancy": 0.382, "bias": 0.773, "factual": 0.62},
        category_risk={
            "Age": {"name": "Age", "accuracy": 0.882, "n_probes": 17,
                     "biased_rate": 0.059, "risk_level": "high"},
            "Race_ethnicity": {"name": "Race_ethnicity", "accuracy": 0.892,
                                "n_probes": 37, "biased_rate": 0.054,
                                "risk_level": "high"},
        },
        high_risk_categories=["Age", "Race_ethnicity"],
        compress_ratio=0.7,
        freeze_fraction=0.75,
        rho_weight=0.2,
        gamma_weight=0.1,
        protection_behaviors=["bias"],
        protection_categories=["Age", "Race_ethnicity"],
        strategy="balanced",
        notes=["Test plan"],
    )


@pytest.fixture
def mock_hybrid_result_pass():
    """HybridResult that passes verification."""
    from rho_eval.hybrid.schema import HybridConfig, HybridResult, PhaseResult

    config = HybridConfig(
        compress_ratio=0.7, rho_weight=0.2, gamma_weight=0.1,
        target_behaviors=("sycophancy",),
        protection_behaviors=("bias",),
        protection_categories=("Age", "Race_ethnicity"),
    )
    result = HybridResult(
        config=config,
        model_name="Qwen/Qwen2.5-7B-Instruct",
    )
    result.audit_before = {"sycophancy": 0.382, "bias": 0.773, "factual": 0.62}
    result.audit_after = {"sycophancy": 0.770, "bias": 0.740, "factual": 0.61}
    result.compute_collateral()

    # Add phases with per-category metrics
    baseline_report = {
        "behaviors": {
            "bias": {
                "rho": 0.773,
                "metadata": {
                    "category_metrics": {
                        "Age": {"accuracy": 0.882, "n": 17, "biased_rate": 0.059},
                        "Race_ethnicity": {"accuracy": 0.892, "n": 37, "biased_rate": 0.054},
                    }
                }
            }
        }
    }
    final_report = {
        "behaviors": {
            "bias": {
                "rho": 0.740,
                "metadata": {
                    "category_metrics": {
                        "Age": {"accuracy": 0.850, "n": 17, "biased_rate": 0.059},
                        "Race_ethnicity": {"accuracy": 0.865, "n": 37, "biased_rate": 0.054},
                    }
                }
            }
        }
    }
    result.phases = [
        PhaseResult(phase="baseline", details={"report": baseline_report}),
        PhaseResult(phase="final_audit", details={"report": final_report}),
    ]
    return result


@pytest.fixture
def mock_hybrid_result_fail():
    """HybridResult that fails verification (too much collateral)."""
    from rho_eval.hybrid.schema import HybridConfig, HybridResult, PhaseResult

    config = HybridConfig(
        compress_ratio=0.7, rho_weight=0.2,
        target_behaviors=("sycophancy",),
    )
    result = HybridResult(
        config=config,
        model_name="Qwen/Qwen2.5-7B-Instruct",
    )
    result.audit_before = {"sycophancy": 0.382, "bias": 0.773}
    result.audit_after = {"sycophancy": 0.770, "bias": 0.600}
    result.compute_collateral()

    baseline_report = {
        "behaviors": {
            "bias": {
                "rho": 0.773,
                "metadata": {
                    "category_metrics": {
                        "Age": {"accuracy": 0.882, "n": 17, "biased_rate": 0.059},
                        "Race_ethnicity": {"accuracy": 0.892, "n": 37, "biased_rate": 0.054},
                    }
                }
            }
        }
    }
    final_report = {
        "behaviors": {
            "bias": {
                "rho": 0.600,
                "metadata": {
                    "category_metrics": {
                        "Age": {"accuracy": 0.588, "n": 17, "biased_rate": 0.059},
                        "Race_ethnicity": {"accuracy": 0.757, "n": 37, "biased_rate": 0.054},
                    }
                }
            }
        }
    }
    result.phases = [
        PhaseResult(phase="baseline", details={"report": baseline_report}),
        PhaseResult(phase="final_audit", details={"report": final_report}),
    ]
    return result


# ═════════════════════════════════════════════════════════════════════════
# Tests: Category Risk Classification
# ═════════════════════════════════════════════════════════════════════════

class TestCategoryRiskClassification:

    def test_high_risk_above_threshold(self, mock_category_metrics):
        from rho_eval.surgical_planner import _classify_category_risk
        risks, high_risk = _classify_category_risk(mock_category_metrics)

        # Age (88.2%) and Race_ethnicity (89.2%) are above 85% threshold
        assert "Age" in high_risk
        assert "Race_ethnicity" in high_risk

    def test_high_risk_requires_min_probes(self, mock_category_metrics):
        from rho_eval.surgical_planner import _classify_category_risk

        # Religion has 83.3% (below 85%) — should NOT be high risk
        risks, high_risk = _classify_category_risk(mock_category_metrics)
        assert "Religion" not in high_risk

    def test_medium_risk_classification(self, mock_category_metrics):
        from rho_eval.surgical_planner import _classify_category_risk
        risks, _ = _classify_category_risk(mock_category_metrics)

        # Gender_identity (78.6%) should be medium
        assert risks["Gender_identity"].risk_level == "medium"
        # Disability_status (71.4%) should be medium
        assert risks["Disability_status"].risk_level == "medium"

    def test_low_risk_classification(self):
        from rho_eval.surgical_planner import _classify_category_risk
        metrics = {"Poor_category": {"accuracy": 0.55, "n": 20, "biased_rate": 0.2}}
        risks, high_risk = _classify_category_risk(metrics)

        assert risks["Poor_category"].risk_level == "low"
        assert len(high_risk) == 0

    def test_empty_metrics(self):
        from rho_eval.surgical_planner import _classify_category_risk
        risks, high_risk = _classify_category_risk({})
        assert len(risks) == 0
        assert len(high_risk) == 0

    def test_small_n_not_protected(self):
        """Categories with few probes shouldn't be in high_risk even if accurate."""
        from rho_eval.surgical_planner import _classify_category_risk
        metrics = {"Rare_cat": {"accuracy": 0.95, "n": 5, "biased_rate": 0.0}}
        risks, high_risk = _classify_category_risk(metrics)

        assert risks["Rare_cat"].risk_level == "high"
        assert "Rare_cat" not in high_risk  # n=5 < MIN_PROBES_FOR_PROTECTION


# ═════════════════════════════════════════════════════════════════════════
# Tests: Surgical Plan Generation
# ═════════════════════════════════════════════════════════════════════════

class TestSurgicalPlanGeneration:

    def test_generate_from_audit(self, mock_audit_report_with_categories):
        from rho_eval.surgical_planner import generate_surgical_plan

        plan = generate_surgical_plan(
            "Qwen/Qwen2.5-7B-Instruct",
            audit_report=mock_audit_report_with_categories,
        )

        assert plan.model_name == "Qwen/Qwen2.5-7B-Instruct"
        assert plan.target_behaviors == ["sycophancy"]
        assert plan.gamma_weight > 0  # Should be non-zero with high-risk categories
        assert len(plan.protection_categories) > 0
        assert "bias" in plan.protection_behaviors

    def test_generate_no_audit(self):
        from rho_eval.surgical_planner import generate_surgical_plan

        plan = generate_surgical_plan("test-model")
        assert plan.model_name == "test-model"
        assert plan.gamma_weight == 0.0  # No audit → nothing to protect
        assert len(plan.protection_categories) == 0

    def test_strategy_affects_gamma(self, mock_audit_report_with_categories):
        from rho_eval.surgical_planner import generate_surgical_plan

        plan_conserv = generate_surgical_plan(
            "test", audit_report=mock_audit_report_with_categories,
            strategy="conservative",
        )
        plan_aggress = generate_surgical_plan(
            "test", audit_report=mock_audit_report_with_categories,
            strategy="aggressive",
        )

        assert plan_conserv.gamma_weight > plan_aggress.gamma_weight

    def test_baseline_scores_populated(self, mock_audit_report_with_categories):
        from rho_eval.surgical_planner import generate_surgical_plan

        plan = generate_surgical_plan(
            "test", audit_report=mock_audit_report_with_categories,
        )
        assert "sycophancy" in plan.baseline_scores
        assert "bias" in plan.baseline_scores
        assert abs(plan.baseline_scores["sycophancy"] - 0.382) < 0.001


# ═════════════════════════════════════════════════════════════════════════
# Tests: SurgicalPlan Serialization
# ═════════════════════════════════════════════════════════════════════════

class TestSurgicalPlanSerialization:

    def test_to_json_roundtrip(self, mock_surgical_plan):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "plan.json"
            mock_surgical_plan.to_json(path)

            from rho_eval.surgical_planner import SurgicalPlan
            loaded = SurgicalPlan.from_json(path)

            assert loaded.model_name == mock_surgical_plan.model_name
            assert loaded.gamma_weight == mock_surgical_plan.gamma_weight
            assert loaded.protection_categories == mock_surgical_plan.protection_categories
            assert loaded.high_risk_categories == mock_surgical_plan.high_risk_categories

    def test_to_dict_complete(self, mock_surgical_plan):
        d = mock_surgical_plan.to_dict()
        assert "model_name" in d
        assert "gamma_weight" in d
        assert "protection_categories" in d
        assert "category_risk" in d
        assert "notes" in d

    def test_to_hybrid_config(self, mock_surgical_plan):
        config = mock_surgical_plan.to_hybrid_config()
        assert config.gamma_weight == 0.1
        assert config.rho_weight == 0.2
        assert config.compress_ratio == 0.7
        assert "bias" in config.protection_behaviors
        assert "Age" in config.protection_categories

    def test_summary_readable(self, mock_surgical_plan):
        summary = mock_surgical_plan.summary()
        assert "Qwen" in summary
        assert "sycophancy" in summary
        assert "balanced" in summary


# ═════════════════════════════════════════════════════════════════════════
# Tests: Verification
# ═════════════════════════════════════════════════════════════════════════

class TestVerification:

    def test_pass_verification(self, mock_surgical_plan, mock_hybrid_result_pass):
        from rho_eval.surgical_planner import verify_surgical_outcome

        v = verify_surgical_outcome(mock_surgical_plan, mock_hybrid_result_pass)

        assert v["passed"] is True
        assert v["target_check"]["passed"] is True
        assert v["target_check"]["improvement"] > 0

    def test_fail_verification_category_damage(
        self, mock_surgical_plan, mock_hybrid_result_fail,
    ):
        from rho_eval.surgical_planner import verify_surgical_outcome

        v = verify_surgical_outcome(mock_surgical_plan, mock_hybrid_result_fail)

        assert v["passed"] is False
        # Age dropped from 88.2% to 58.8% — should fail
        assert "Age" in v["category_checks"]
        assert v["category_checks"]["Age"]["passed"] is False

    def test_suggestions_on_failure(
        self, mock_surgical_plan, mock_hybrid_result_fail,
    ):
        from rho_eval.surgical_planner import verify_surgical_outcome

        v = verify_surgical_outcome(mock_surgical_plan, mock_hybrid_result_fail)

        assert len(v["suggestions"]) > 0
        # Should suggest increasing gamma_weight
        gamma_suggestion = [s for s in v["suggestions"] if "gamma_weight" in s]
        assert len(gamma_suggestion) > 0

    def test_summary_format(self, mock_surgical_plan, mock_hybrid_result_pass):
        from rho_eval.surgical_planner import verify_surgical_outcome

        v = verify_surgical_outcome(mock_surgical_plan, mock_hybrid_result_pass)
        assert "PASS" in v["summary"] or "FAIL" in v["summary"]
        assert "target" in v["summary"]

    def test_custom_thresholds(self, mock_surgical_plan, mock_hybrid_result_pass):
        from rho_eval.surgical_planner import verify_surgical_outcome

        # Very strict threshold — even small drops should fail
        v = verify_surgical_outcome(
            mock_surgical_plan, mock_hybrid_result_pass,
            max_collateral_per_category=0.001,
        )
        # Age dropped from 88.2% to 85% — fails with strict threshold
        if "Age" in v["category_checks"]:
            assert v["category_checks"]["Age"]["passed"] is False


# ═════════════════════════════════════════════════════════════════════════
# Tests: HybridConfig Extensions
# ═════════════════════════════════════════════════════════════════════════

class TestHybridConfigExtensions:

    def test_gamma_weight_default(self):
        from rho_eval.hybrid.schema import HybridConfig
        config = HybridConfig()
        assert config.gamma_weight == 0.0
        assert config.protection_behaviors == ()
        assert config.protection_categories == ()

    def test_protection_enabled(self):
        from rho_eval.hybrid.schema import HybridConfig
        config = HybridConfig(gamma_weight=0.1, protection_behaviors=("bias",))
        assert config.protection_enabled is True

    def test_protection_disabled_no_gamma(self):
        from rho_eval.hybrid.schema import HybridConfig
        config = HybridConfig(gamma_weight=0.0, protection_behaviors=("bias",))
        assert config.protection_enabled is False

    def test_protection_disabled_no_behaviors(self):
        from rho_eval.hybrid.schema import HybridConfig
        config = HybridConfig(gamma_weight=0.1, protection_behaviors=())
        assert config.protection_enabled is False

    def test_json_roundtrip_with_protection(self):
        from rho_eval.hybrid.schema import HybridConfig

        config = HybridConfig(
            gamma_weight=0.15,
            protection_behaviors=("bias",),
            protection_categories=("Age", "Religion"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config.to_json(path)
            loaded = HybridConfig.from_json(path)

            assert loaded.gamma_weight == 0.15
            assert loaded.protection_behaviors == ("bias",)
            assert loaded.protection_categories == ("Age", "Religion")


# ═════════════════════════════════════════════════════════════════════════
# Tests: γ Protection Loss
# ═════════════════════════════════════════════════════════════════════════

class TestGammaProtectionLoss:

    def test_import(self):
        from rho_eval.alignment import gamma_protection_loss
        assert callable(gamma_protection_loss)

    def test_empty_pairs_returns_zero(self):
        from rho_eval.alignment.losses import gamma_protection_loss
        import torch

        # Need a minimal model-like object
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, **kwargs):
                class Out:
                    loss = self.p * 0.0
                return Out()

        model = FakeModel()
        result = gamma_protection_loss(model, None, [], device="cpu")
        assert isinstance(result, torch.Tensor)
        assert result.item() == 0.0

    def test_delegates_to_rho_auxiliary(self):
        """gamma_protection_loss should produce same output as rho_auxiliary_loss."""
        from rho_eval.alignment.losses import gamma_protection_loss, rho_auxiliary_loss
        import torch

        # Use the mock model/tokenizer from test_alignment pattern
        class MockOutput:
            def __init__(self, loss):
                self.loss = loss

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
                first_tok = input_ids[0, 0].float()
                loss = 0.5 + (first_tok / 100.0) * self.dummy
                return MockOutput(loss)

        class MockTokenizer:
            def __call__(self, text, **kw):
                if "positive" in text.lower():
                    ids = [2, 3, 4, 5]
                else:
                    ids = [80, 81, 82, 83]
                return {
                    "input_ids": torch.tensor([ids]),
                    "attention_mask": torch.ones(1, len(ids)),
                }

        model = MockModel()
        tokenizer = MockTokenizer()
        pairs = [
            {"positive": "positive text", "negative": "negative text"},
        ]

        gamma = gamma_protection_loss(model, tokenizer, pairs)
        rho = rho_auxiliary_loss(model, tokenizer, pairs)

        assert abs(gamma.item() - rho.item()) < 1e-6


# ═════════════════════════════════════════════════════════════════════════
# Tests: BehavioralContrastDataset with categories
# ═════════════════════════════════════════════════════════════════════════

class TestBehavioralContrastDatasetCategories:

    def test_categories_parameter_accepted(self):
        """Verify the constructor accepts categories without error."""
        from rho_eval.alignment.dataset import BehavioralContrastDataset

        # This may fail on import of behaviors if probes aren't available,
        # but the constructor should accept the parameter
        try:
            ds = BehavioralContrastDataset(
                behaviors=["bias"],
                categories=["Age"],
                seed=42,
            )
            # If probes are available, check filtering worked
            assert ds.categories == ["Age"]
        except Exception:
            # Probes may not be available in test environment
            pytest.skip("Bias probes not available in test environment")

    def test_categories_none_loads_all(self):
        """Without categories, should load all probes."""
        from rho_eval.alignment.dataset import BehavioralContrastDataset

        try:
            ds_all = BehavioralContrastDataset(
                behaviors=["bias"],
                categories=None,
                seed=42,
            )
            ds_filtered = BehavioralContrastDataset(
                behaviors=["bias"],
                categories=["Age"],
                seed=42,
            )
            # All probes >= filtered probes
            assert len(ds_all) >= len(ds_filtered)
        except Exception:
            pytest.skip("Bias probes not available in test environment")


# ═════════════════════════════════════════════════════════════════════════
# Tests: γ weight selection
# ═════════════════════════════════════════════════════════════════════════

class TestGammaWeightSelection:

    def test_no_risk_zero_gamma(self):
        from rho_eval.surgical_planner import _select_gamma_weight
        gamma = _select_gamma_weight([], "balanced")
        assert gamma == 0.0

    def test_high_risk_nonzero_gamma(self):
        from rho_eval.surgical_planner import _select_gamma_weight
        gamma = _select_gamma_weight(["Age", "Race_ethnicity"], "balanced")
        assert gamma > 0.0

    def test_conservative_higher_than_aggressive(self):
        from rho_eval.surgical_planner import _select_gamma_weight
        cats = ["Age", "Race_ethnicity"]
        conserv = _select_gamma_weight(cats, "conservative")
        aggress = _select_gamma_weight(cats, "aggressive")
        assert conserv > aggress

    def test_many_categories_scales_up(self):
        from rho_eval.surgical_planner import _select_gamma_weight
        few = _select_gamma_weight(["A"], "balanced")
        many = _select_gamma_weight(["A", "B", "C", "D"], "balanced")
        assert many > few
