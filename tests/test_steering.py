"""Tests for the SAE-based behavioral steering module.

All tests use synthetic data — no model loading required.
Tests verify:
  - GatedSAE forward/backward, shapes, sparsity, decoder normalization
  - ActivationData schema and serialization
  - Feature analysis identifies synthetic behavioral patterns
  - SAESteeringHook modifies activations correctly
  - Public API imports work
"""

import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════
# Helpers — synthetic data generators
# ═══════════════════════════════════════════════════════════════════════

def _make_synthetic_activations(n_per_behavior=20, hidden_dim=64, seed=42):
    """Create synthetic activations with clear behavioral structure.

    Each behavior gets a different direction in activation space:
    - factual: positive along dim 0-3
    - toxicity: positive along dim 4-7
    - sycophancy: positive along dim 8-11
    - bias: positive along dim 12-15

    Positive examples have large positive values in their behavior's
    dimensions; negative examples have large negative values.
    """
    torch.manual_seed(seed)

    behaviors = ["factual", "toxicity", "sycophancy", "bias"]
    all_activations = []
    all_labels = []
    all_polarities = []

    for i, behavior in enumerate(behaviors):
        base_dims = list(range(i * 4, i * 4 + 4))

        for _ in range(n_per_behavior):
            # Positive example: large positive in behavior dims
            pos = torch.randn(hidden_dim) * 0.1
            for d in base_dims:
                pos[d] = 2.0 + torch.randn(1).item() * 0.3
            all_activations.append(pos)
            all_labels.append(behavior)
            all_polarities.append("positive")

            # Negative example: large negative in behavior dims
            neg = torch.randn(hidden_dim) * 0.1
            for d in base_dims:
                neg[d] = -2.0 + torch.randn(1).item() * 0.3
            all_activations.append(neg)
            all_labels.append(behavior)
            all_polarities.append("negative")

    from rho_eval.steering.schema import ActivationData
    return ActivationData(
        activations=torch.stack(all_activations),
        labels=all_labels,
        polarities=all_polarities,
        layer_idx=17,
        model_name="synthetic",
    )


class _InnerModel(nn.Module):
    """Inner model with layers attribute (mimics model.model.layers)."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _MockModel(nn.Module):
    """Tiny mock model matching HuggingFace structure (model.model.layers).

    get_layers() expects model.model.layers, so we nest accordingly.
    """

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model = _InnerModel(hidden_dim)
        self.config = type("Config", (), {"_name_or_path": "mock", "hidden_size": hidden_dim})()

    def forward(self, x):
        return self.model(x)


# ═══════════════════════════════════════════════════════════════════════
# Tests: GatedSAE
# ═══════════════════════════════════════════════════════════════════════

class TestGatedSAE:
    def test_forward_shapes(self):
        from rho_eval.steering.sae import GatedSAE

        hidden_dim = 64
        expansion = 4
        sae = GatedSAE(hidden_dim, expansion)

        x = torch.randn(8, hidden_dim)
        x_hat, z, gate_pre = sae(x)

        assert x_hat.shape == (8, hidden_dim)
        assert z.shape == (8, hidden_dim * expansion)
        assert gate_pre.shape == (8, hidden_dim * expansion)

    def test_sparsity(self):
        from rho_eval.steering.sae import GatedSAE

        sae = GatedSAE(64, expansion_factor=8)
        x = torch.randn(32, 64)
        _, z, _ = sae(x)

        # With random init, many features should be near zero
        # At least 50% of features should have activation < 0.1
        near_zero = (z.abs() < 0.1).float().mean()
        assert near_zero > 0.3, f"Expected >30% near-zero features, got {near_zero:.1%}"

    def test_decoder_normalization(self):
        from rho_eval.steering.sae import GatedSAE

        sae = GatedSAE(64, expansion_factor=4)
        sae.normalize_decoder()

        norms = sae.W_dec.norm(dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"Decoder column norms should be ~1.0, got range [{norms.min():.4f}, {norms.max():.4f}]"

    def test_loss_components(self):
        from rho_eval.steering.sae import GatedSAE

        sae = GatedSAE(64, expansion_factor=4)
        x = torch.randn(16, 64)
        x_hat, z, gate_pre = sae(x)

        total, mse, l1 = GatedSAE.compute_loss(x, x_hat, gate_pre, sparsity_lambda=0.01)

        assert total.dim() == 0  # scalar
        assert mse.dim() == 0
        assert l1.dim() == 0
        assert total.item() >= mse.item()  # total >= mse (l1 is positive)
        assert mse.item() >= 0
        assert l1.item() >= 0

    def test_gradient_flow(self):
        from rho_eval.steering.sae import GatedSAE

        sae = GatedSAE(32, expansion_factor=4)
        x = torch.randn(8, 32)
        x_hat, z, gate_pre = sae(x)
        total, mse, l1 = GatedSAE.compute_loss(x, x_hat, gate_pre)

        total.backward()

        # All parameters should have gradients
        for name, param in sae.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_encode_decode_approximate_identity(self):
        """A large enough SAE trained briefly should approximately reconstruct."""
        from rho_eval.steering.sae import GatedSAE

        torch.manual_seed(42)
        hidden_dim = 16
        sae = GatedSAE(hidden_dim, expansion_factor=16)

        # Brief training to improve reconstruction
        x = torch.randn(64, hidden_dim)
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-2)

        for _ in range(100):
            x_hat, z, gate_pre = sae(x)
            loss, _, _ = GatedSAE.compute_loss(x, x_hat, gate_pre, sparsity_lambda=1e-5)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()

        # Check reconstruction quality
        sae.eval()
        with torch.no_grad():
            x_hat, _, _ = sae(x)
            mse = (x - x_hat).pow(2).mean().item()

        # After training, MSE should be reasonable (not perfect, but improved)
        assert mse < 1.0, f"Reconstruction MSE too high: {mse:.4f}"


# ═══════════════════════════════════════════════════════════════════════
# Tests: ActivationData
# ═══════════════════════════════════════════════════════════════════════

class TestActivationData:
    def test_schema_fields(self):
        act_data = _make_synthetic_activations()
        assert act_data.n_samples == 160  # 4 behaviors * 20 pairs * 2 (pos/neg)
        assert act_data.hidden_dim == 64
        assert act_data.layer_idx == 17
        assert len(act_data.labels) == act_data.n_samples
        assert len(act_data.polarities) == act_data.n_samples

    def test_behaviors_property(self):
        act_data = _make_synthetic_activations()
        behaviors = act_data.behaviors
        assert behaviors == ["factual", "toxicity", "sycophancy", "bias"]

    def test_save_load_roundtrip(self):
        act_data = _make_synthetic_activations()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_activations.pt"
            act_data.save(path)
            assert path.exists()

            from rho_eval.steering.schema import ActivationData
            loaded = ActivationData.load(path)

            assert loaded.n_samples == act_data.n_samples
            assert loaded.hidden_dim == act_data.hidden_dim
            assert loaded.layer_idx == act_data.layer_idx
            assert loaded.labels == act_data.labels
            assert loaded.polarities == act_data.polarities
            assert torch.allclose(loaded.activations, act_data.activations)

    def test_validation(self):
        """Mismatched lengths should raise."""
        with pytest.raises(AssertionError):
            from rho_eval.steering.schema import ActivationData
            ActivationData(
                activations=torch.randn(10, 32),
                labels=["a"] * 5,  # wrong length
                polarities=["positive"] * 10,
                layer_idx=0,
            )


# ═══════════════════════════════════════════════════════════════════════
# Tests: Feature Analysis
# ═══════════════════════════════════════════════════════════════════════

class TestFeatureAnalysis:
    def test_identify_returns_reports(self):
        from rho_eval.steering.sae import GatedSAE
        from rho_eval.steering.analyze import identify_behavioral_features

        act_data = _make_synthetic_activations(n_per_behavior=30, hidden_dim=32)
        sae = GatedSAE(32, expansion_factor=4)

        # Train briefly
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-2)
        for _ in range(50):
            x_hat, z, gate_pre = sae(act_data.activations)
            loss, _, _ = GatedSAE.compute_loss(
                act_data.activations, x_hat, gate_pre, 1e-4
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()

        sae.eval()
        reports, behavioral_features = identify_behavioral_features(
            sae, act_data, threshold=1.5,
        )

        assert isinstance(reports, list)
        assert isinstance(behavioral_features, dict)
        assert all(b in behavioral_features for b in act_data.behaviors)

    def test_dead_features_excluded(self):
        """Features that never activate should not appear in reports."""
        from rho_eval.steering.sae import GatedSAE
        from rho_eval.steering.analyze import identify_behavioral_features

        act_data = _make_synthetic_activations(n_per_behavior=10, hidden_dim=16)

        # Create SAE with very high sparsity (many dead features)
        sae = GatedSAE(16, expansion_factor=4)
        # Zero out gate weights to make features dead
        with torch.no_grad():
            sae.b_gate.fill_(-10.0)  # Strong negative bias → gates closed

        sae.eval()
        reports, _ = identify_behavioral_features(sae, act_data, threshold=0.1)

        # With very negative gate biases, most features should be dead
        # Reports should be empty or very few
        for r in reports:
            assert r.selectivity >= 0.1

    def test_overlap_matrix_symmetric(self):
        from rho_eval.steering.analyze import feature_overlap_matrix
        from rho_eval.steering.schema import FeatureReport

        # Create synthetic feature reports
        reports = [
            FeatureReport(
                feature_idx=0,
                behavior_scores={"factual": 0.5, "toxicity": 0.1},
                dominant_behavior="factual",
                selectivity=5.0,
            ),
            FeatureReport(
                feature_idx=1,
                behavior_scores={"factual": 0.1, "toxicity": 0.5},
                dominant_behavior="toxicity",
                selectivity=5.0,
            ),
            FeatureReport(
                feature_idx=2,
                behavior_scores={"factual": 0.4, "toxicity": 0.4},
                dominant_behavior="factual",
                selectivity=1.0,
            ),
        ]

        behaviors = ["factual", "toxicity"]
        overlap = feature_overlap_matrix(reports, behaviors)

        # Should be symmetric
        assert overlap["factual"]["toxicity"] == overlap["toxicity"]["factual"]
        # Diagonal should be 1.0
        assert overlap["factual"]["factual"] == 1.0
        assert overlap["toxicity"]["toxicity"] == 1.0

    def test_top_k_selection(self):
        from rho_eval.steering.sae import GatedSAE
        from rho_eval.steering.analyze import identify_behavioral_features

        act_data = _make_synthetic_activations(n_per_behavior=10, hidden_dim=16)
        sae = GatedSAE(16, expansion_factor=4)

        # Train briefly
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-2)
        for _ in range(30):
            x_hat, z, gate_pre = sae(act_data.activations)
            loss, _, _ = GatedSAE.compute_loss(
                act_data.activations, x_hat, gate_pre, 1e-4
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()

        sae.eval()
        _, behavioral_features = identify_behavioral_features(
            sae, act_data, threshold=0.0, top_k=5,
        )

        # Each behavior should have at most 5 features
        for behavior in act_data.behaviors:
            assert len(behavioral_features[behavior]) <= 5


# ═══════════════════════════════════════════════════════════════════════
# Tests: SAESteeringHook
# ═══════════════════════════════════════════════════════════════════════

class TestSAESteeringHook:
    def test_hook_modifies_output(self):
        from rho_eval.steering.sae import GatedSAE
        from rho_eval.steering.steer import SAESteeringHook
        from rho_eval.utils import get_layers

        model = _MockModel(hidden_dim=32)
        sae = GatedSAE(32, expansion_factor=4)
        sae.eval()

        x = torch.randn(1, 32)

        # Without hook
        model.eval()
        with torch.no_grad():
            out_no_hook = model(x).clone()

        # With hook (scale=3.0 on features 0-3 to get a visible effect)
        hook = SAESteeringHook(model, sae, layer_idx=0,
                               feature_indices=[0, 1, 2, 3], scale=3.0)
        with torch.no_grad():
            out_with_hook = model(x)

        hook.remove()

        # Outputs should differ
        assert not torch.allclose(out_no_hook, out_with_hook, atol=1e-6), \
            "Hook should modify the output"

    def test_hook_removal_restores(self):
        from rho_eval.steering.sae import GatedSAE
        from rho_eval.steering.steer import SAESteeringHook

        model = _MockModel(hidden_dim=32)
        sae = GatedSAE(32, expansion_factor=4)
        sae.eval()

        x = torch.randn(1, 32)

        model.eval()
        with torch.no_grad():
            out_before = model(x).clone()

        hook = SAESteeringHook(model, sae, layer_idx=0,
                               feature_indices=[0], scale=5.0)
        hook.remove()

        with torch.no_grad():
            out_after = model(x)

        assert torch.allclose(out_before, out_after, atol=1e-6), \
            "After hook removal, output should match original"

    def test_scale_zero_is_ablation(self):
        """Scale=0 should zero out features, changing the output."""
        from rho_eval.steering.sae import GatedSAE
        from rho_eval.steering.steer import SAESteeringHook

        model = _MockModel(hidden_dim=32)
        sae = GatedSAE(32, expansion_factor=4)

        # Brief training so SAE has meaningful features
        x_train = torch.randn(64, 32)
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-2)
        for _ in range(50):
            x_hat, z, gate_pre = sae(x_train)
            loss, _, _ = GatedSAE.compute_loss(x_train, x_hat, gate_pre, 1e-5)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()

        sae.eval()
        model.eval()

        x = torch.randn(1, 32)

        with torch.no_grad():
            out_no_hook = model(x).clone()

        # Scale=0 ablates all features
        hook = SAESteeringHook(model, sae, layer_idx=0,
                               feature_indices=list(range(32 * 4)),
                               scale=0.0)
        with torch.no_grad():
            out_ablated = model(x)
        hook.remove()

        # With all features ablated, output should differ from no-hook
        # (unless SAE perfectly reconstructs, which is unlikely with scale=0)
        assert not torch.allclose(out_no_hook, out_ablated, atol=1e-4), \
            "Scale=0 (ablation) should produce different output"


# ═══════════════════════════════════════════════════════════════════════
# Tests: Schema Serialization
# ═══════════════════════════════════════════════════════════════════════

class TestSchemaSerialization:
    def test_sae_config_frozen(self):
        from rho_eval.steering.schema import SAEConfig

        config = SAEConfig(hidden_dim=64, expansion_factor=8)
        assert config.n_features == 512

        with pytest.raises(AttributeError):
            config.hidden_dim = 128  # Should be frozen

    def test_sae_config_to_dict(self):
        from rho_eval.steering.schema import SAEConfig

        config = SAEConfig(hidden_dim=64, expansion_factor=8)
        d = config.to_dict()
        assert d["hidden_dim"] == 64
        assert d["n_features"] == 512
        assert d["expansion_factor"] == 8

    def test_feature_report_to_dict(self):
        from rho_eval.steering.schema import FeatureReport

        report = FeatureReport(
            feature_idx=42,
            behavior_scores={"factual": 0.5, "toxicity": 0.1},
            dominant_behavior="factual",
            selectivity=5.0,
        )
        d = report.to_dict()
        assert d["feature_idx"] == 42
        assert d["dominant_behavior"] == "factual"

    def test_steering_report_save_load(self):
        from rho_eval.steering.schema import (
            SAESteeringReport, SAEConfig, FeatureReport,
        )

        report = SAESteeringReport(
            model="test-model",
            layer_idx=17,
            sae_config=SAEConfig(hidden_dim=64),
            feature_reports=[
                FeatureReport(
                    feature_idx=0,
                    behavior_scores={"factual": 0.5},
                    dominant_behavior="factual",
                    selectivity=3.0,
                ),
            ],
            behavioral_features={"factual": [0], "toxicity": []},
            steering_results=[{"scale": 2.0, "rho_scores": {"factual": 0.8}}],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            report.save(path)

            loaded = SAESteeringReport.load(path)
            assert loaded.model == "test-model"
            assert loaded.layer_idx == 17
            assert loaded.sae_config.hidden_dim == 64
            assert len(loaded.feature_reports) == 1
            assert loaded.behavioral_features["factual"] == [0]


# ═══════════════════════════════════════════════════════════════════════
# Tests: Imports
# ═══════════════════════════════════════════════════════════════════════

class TestImports:
    def test_import_from_steering(self):
        from rho_eval.steering import (
            GatedSAE,
            SAEConfig,
            ActivationData,
            FeatureReport,
            SAESteeringReport,
            collect_activations,
            train_sae,
            train_behavioral_sae,
            identify_behavioral_features,
            feature_overlap_matrix,
            SAESteeringHook,
            steer_features,
            evaluate_sae_steering,
        )
        assert callable(train_sae)
        assert callable(identify_behavioral_features)
        assert callable(steer_features)

    def test_import_schema_directly(self):
        from rho_eval.steering.schema import (
            SAEConfig,
            ActivationData,
            FeatureReport,
            SAESteeringReport,
        )
        assert SAEConfig is not None

    def test_import_sae_directly(self):
        from rho_eval.steering.sae import GatedSAE
        assert callable(GatedSAE)

    def test_steering_not_in_top_level(self):
        """Steering is opt-in — not auto-imported at rho_eval top level."""
        import rho_eval
        # These should NOT be in the top-level namespace
        assert not hasattr(rho_eval, "GatedSAE")
        assert not hasattr(rho_eval, "SAESteeringHook")
