"""Tests for the interpretability module.

All tests use synthetic tensors — no model loading required.
"""

import json
import math
import tempfile
from pathlib import Path

import pytest
import torch


# ═══════════════════════════════════════════════════════════════════════════
# Schema Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSubspaceResult:
    """Test SubspaceResult dataclass."""

    def test_to_dict(self, mock_subspace_result):
        d = mock_subspace_result.to_dict()
        assert d["behavior"] == "test"
        assert d["layer_idx"] == 17
        assert d["n_pairs"] == 50
        assert d["effective_dim"] == 3
        assert "singular_values" in d
        assert "explained_variance" in d
        assert len(d["singular_values"]) <= 20  # truncated

    def test_directions_shape(self, mock_subspace_result):
        assert mock_subspace_result.directions.shape[0] == 10
        assert mock_subspace_result.directions.shape[1] == 64

    def test_directions_approximately_orthogonal(self, mock_subspace_result):
        D = mock_subspace_result.directions.float()
        gram = D @ D.T  # Should be close to identity
        eye = torch.eye(D.shape[0])
        diff = (gram - eye).abs().max().item()
        assert diff < 0.01, f"Directions not orthogonal: max deviation = {diff}"

    def test_explained_variance_monotonic(self, mock_subspace_result):
        ev = mock_subspace_result.explained_variance
        for i in range(1, len(ev)):
            assert ev[i] >= ev[i - 1], f"Variance not monotonic at {i}"

    def test_explained_variance_bounded(self, mock_subspace_result):
        ev = mock_subspace_result.explained_variance
        assert ev[0] >= 0.0
        assert ev[-1] <= 1.0 + 1e-6


class TestOverlapMatrix:
    """Test OverlapMatrix dataclass."""

    def test_to_dict(self, mock_overlap_matrix):
        d = mock_overlap_matrix.to_dict()
        assert d["layer_idx"] == 17
        assert len(d["behaviors"]) == 3
        assert len(d["cosine_matrix"]) == 3
        assert len(d["cosine_matrix"][0]) == 3

    def test_diagonal_is_one(self, mock_overlap_matrix):
        for i in range(3):
            assert mock_overlap_matrix.cosine_matrix[i][i] == 1.0
            assert mock_overlap_matrix.shared_variance[i][i] == 1.0
            assert mock_overlap_matrix.subspace_angles[i][i] == 0.0

    def test_symmetric(self, mock_overlap_matrix):
        n = len(mock_overlap_matrix.behaviors)
        for i in range(n):
            for j in range(n):
                assert abs(mock_overlap_matrix.cosine_matrix[i][j] -
                           mock_overlap_matrix.cosine_matrix[j][i]) < 1e-6


class TestSurgicalResult:
    """Test SurgicalResult dataclass."""

    def test_to_dict(self):
        from rho_eval.interpretability.schema import SurgicalResult
        sr = SurgicalResult(
            intervention="rank_1",
            target_behavior="sycophancy",
            rho_scores={"factual": 0.5, "sycophancy": 0.3, "bias": 0.7},
            config={"layer": 17, "alpha": 4.0, "rank": 1},
        )
        d = sr.to_dict()
        assert d["intervention"] == "rank_1"
        assert d["rho_scores"]["factual"] == 0.5
        assert d["config"]["layer"] == 17


class TestInterpretabilityReport:
    """Test InterpretabilityReport save/load round-trip."""

    def test_create_empty(self):
        from rho_eval.interpretability.schema import InterpretabilityReport
        r = InterpretabilityReport(model="test-model")
        assert r.model == "test-model"
        assert len(r.subspaces) == 0
        assert len(r.overlaps) == 0

    def test_to_dict(self, mock_subspace_result, mock_overlap_matrix):
        from rho_eval.interpretability.schema import InterpretabilityReport
        r = InterpretabilityReport(model="test-model")
        r.subspaces["test"] = {17: mock_subspace_result}
        r.overlaps[17] = mock_overlap_matrix
        d = r.to_dict()
        assert d["model"] == "test-model"
        assert "17" in d["subspaces"]["test"]
        assert "17" in d["overlaps"]

    def test_json_roundtrip(self, mock_subspace_result, mock_overlap_matrix):
        from rho_eval.interpretability.schema import InterpretabilityReport, SurgicalResult

        report = InterpretabilityReport(model="test-model")
        report.subspaces["test"] = {17: mock_subspace_result}
        report.overlaps[17] = mock_overlap_matrix
        report.surgical_results.append(SurgicalResult(
            intervention="rank_1",
            target_behavior="test",
            rho_scores={"test": 0.5},
            config={"layer": 17},
        ))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            pt_path = f.name

        report.save(json_path)
        report.save_tensors(pt_path)

        loaded = InterpretabilityReport.load(json_path, pt_path)

        assert loaded.model == "test-model"
        assert "test" in loaded.subspaces
        assert 17 in loaded.subspaces["test"]
        assert 17 in loaded.overlaps
        assert len(loaded.surgical_results) == 1

        # Check tensor data was preserved
        orig_dir = report.subspaces["test"][17].directions
        loaded_dir = loaded.subspaces["test"][17].directions
        assert torch.allclose(orig_dir, loaded_dir, atol=1e-5)

        Path(json_path).unlink()
        Path(pt_path).unlink()

    def test_repr(self, mock_subspace_result, mock_overlap_matrix):
        from rho_eval.interpretability.schema import InterpretabilityReport
        r = InterpretabilityReport(model="test")
        r.subspaces["a"] = {17: mock_subspace_result}
        r.overlaps[17] = mock_overlap_matrix
        s = repr(r)
        assert "test" in s
        assert "behaviors=1" in s


# ═══════════════════════════════════════════════════════════════════════════
# Overlap Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestOverlapComputation:
    """Test overlap computation with synthetic subspaces."""

    def _make_subspace(self, directions, behavior="test", layer=17):
        from rho_eval.interpretability.schema import SubspaceResult
        hidden_dim = directions.shape[1]
        return SubspaceResult(
            behavior=behavior,
            layer_idx=layer,
            n_pairs=50,
            directions=directions,
            singular_values=[1.0] * directions.shape[0],
            explained_variance=[1.0],
            effective_dim=1,
            mean_pos=torch.zeros(hidden_dim),
            mean_neg=torch.zeros(hidden_dim),
            steering_vector=torch.zeros(hidden_dim),
        )

    def test_identical_subspaces(self):
        from rho_eval.interpretability.overlap import compute_overlap

        # Create properly orthonormal directions via QR
        torch.manual_seed(42)
        A = torch.randn(64, 5)
        Q, _ = torch.linalg.qr(A)
        D = Q[:, :5].T  # (5, 64) orthonormal

        sr1 = self._make_subspace(D, "a", 17)
        sr2 = self._make_subspace(D, "b", 17)

        subspaces = {"a": {17: sr1}, "b": {17: sr2}}
        overlaps = compute_overlap(subspaces, top_k=5, verbose=False)

        om = overlaps[17]
        # Off-diagonal should be ~1.0
        assert abs(om.cosine_matrix[0][1]) > 0.99
        assert om.shared_variance[0][1] > 0.99
        assert om.subspace_angles[0][1] < 5.0  # Near 0 degrees

    def test_orthogonal_subspaces(self):
        from rho_eval.interpretability.overlap import compute_overlap

        # Create two orthogonal sets of directions
        torch.manual_seed(42)
        A = torch.randn(64, 64)
        Q, _ = torch.linalg.qr(A)

        D1 = Q[:5].clone()   # First 5 orthogonal directions
        D2 = Q[5:10].clone()  # Next 5 orthogonal directions

        sr1 = self._make_subspace(D1, "a", 17)
        sr2 = self._make_subspace(D2, "b", 17)

        subspaces = {"a": {17: sr1}, "b": {17: sr2}}
        overlaps = compute_overlap(subspaces, top_k=5, verbose=False)

        om = overlaps[17]
        # Should be near-zero overlap
        assert abs(om.cosine_matrix[0][1]) < 0.1
        assert om.shared_variance[0][1] < 0.1
        assert om.subspace_angles[0][1] > 80.0  # Near 90 degrees

    def test_overlap_is_symmetric(self):
        from rho_eval.interpretability.overlap import compute_overlap

        torch.manual_seed(42)
        D1 = torch.randn(5, 64)
        D2 = torch.randn(5, 64)

        sr1 = self._make_subspace(D1, "a", 17)
        sr2 = self._make_subspace(D2, "b", 17)

        subspaces = {"a": {17: sr1}, "b": {17: sr2}}
        overlaps = compute_overlap(subspaces, top_k=5, verbose=False)

        om = overlaps[17]
        assert abs(om.cosine_matrix[0][1] - om.cosine_matrix[1][0]) < 1e-4
        assert abs(om.shared_variance[0][1] - om.shared_variance[1][0]) < 1e-4
        assert abs(om.subspace_angles[0][1] - om.subspace_angles[1][0]) < 0.01

    def test_self_overlap_is_one(self):
        from rho_eval.interpretability.overlap import compute_overlap

        torch.manual_seed(42)
        D1 = torch.randn(5, 64)
        D2 = torch.randn(5, 64)

        sr1 = self._make_subspace(D1, "a", 17)
        sr2 = self._make_subspace(D2, "b", 17)

        subspaces = {"a": {17: sr1}, "b": {17: sr2}}
        overlaps = compute_overlap(subspaces, top_k=5, verbose=False)

        om = overlaps[17]
        assert om.cosine_matrix[0][0] == 1.0
        assert om.cosine_matrix[1][1] == 1.0
        assert om.subspace_angles[0][0] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Surgical Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestOrthogonalProject:
    """Test orthogonal projection functions."""

    def test_removes_component(self, mock_subspace_result):
        from rho_eval.interpretability.surgical import orthogonal_project

        # Create a vector that has a known component in the subspace
        v = mock_subspace_result.directions[0].clone()  # Lies entirely in subspace
        v_clean = orthogonal_project(v, mock_subspace_result, n_directions=1)

        # Should be nearly zero (all of v was in the removed subspace)
        assert v_clean.norm().item() < 0.01

    def test_preserves_orthogonal_component(self, mock_subspace_result):
        from rho_eval.interpretability.surgical import orthogonal_project

        # Create a vector orthogonal to the first direction
        D = mock_subspace_result.directions
        v = torch.randn(D.shape[1])
        # Remove component along first direction
        d0 = D[0].float()
        v = v - (torch.dot(v.float(), d0) / torch.dot(d0, d0)) * d0

        v_clean = orthogonal_project(v, mock_subspace_result, n_directions=1)

        # Should be approximately the same vector
        cos_sim = torch.dot(v.float(), v_clean.float()) / (v.float().norm() * v_clean.float().norm())
        assert cos_sim.item() > 0.99

    def test_reduces_norm(self, mock_subspace_result):
        from rho_eval.interpretability.surgical import orthogonal_project

        v = torch.randn(mock_subspace_result.directions.shape[1])
        v_clean = orthogonal_project(v, mock_subspace_result, n_directions=5)

        # Cleaned vector should have smaller or equal norm
        assert v_clean.norm().item() <= v.norm().item() + 1e-6

    def test_result_is_orthogonal(self, mock_subspace_result):
        from rho_eval.interpretability.surgical import orthogonal_project

        v = torch.randn(mock_subspace_result.directions.shape[1])
        v_clean = orthogonal_project(v, mock_subspace_result, n_directions=3)

        # Result should be orthogonal to removed directions
        for i in range(3):
            d = mock_subspace_result.directions[i].float()
            dot = abs(torch.dot(v_clean.float(), d).item())
            assert dot < 0.01, f"Not orthogonal to direction {i}: dot={dot}"


class TestRankKSteer:
    """Test rank-k steering."""

    def test_rank_1_preserves_direction(self, mock_subspace_result):
        from rho_eval.interpretability.surgical import rank_k_steer

        v_k = rank_k_steer(mock_subspace_result, rank=1)
        d0 = mock_subspace_result.directions[0].float()

        # v_k should be parallel to the first direction
        cos_sim = torch.dot(v_k, d0) / (v_k.norm() * d0.norm())
        assert abs(cos_sim.item()) > 0.99

    def test_rank_increases_fidelity(self, mock_subspace_result):
        from rho_eval.interpretability.surgical import rank_k_steer

        full_v = mock_subspace_result.steering_vector.float()

        v1 = rank_k_steer(mock_subspace_result, rank=1)
        v5 = rank_k_steer(mock_subspace_result, rank=5)

        # Higher rank should be closer to full vector
        err1 = (full_v - v1).norm().item()
        err5 = (full_v - v5).norm().item()
        assert err5 <= err1 + 1e-6

    def test_full_rank_recovers_projection(self, mock_subspace_result):
        from rho_eval.interpretability.surgical import rank_k_steer

        n = mock_subspace_result.directions.shape[0]
        v_full = rank_k_steer(mock_subspace_result, rank=n)

        # This should be the projection of steering_vector onto the subspace
        # It won't equal steering_vector unless the vector is entirely in the subspace
        assert v_full.norm().item() > 0


# ═══════════════════════════════════════════════════════════════════════════
# Contrast Pair Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildContrastPairs:
    """Test contrast pair construction."""

    def test_factual_pairs(self, mock_probes):
        from rho_eval.interpretability.activation import build_contrast_pairs
        pairs = build_contrast_pairs("factual", mock_probes)
        assert len(pairs) == 3
        assert "positive" in pairs[0]
        assert "negative" in pairs[0]
        assert pairs[0]["positive"] != pairs[0]["negative"]

    def test_toxicity_pairs(self, mock_toxicity_probes):
        from rho_eval.interpretability.activation import build_contrast_pairs
        pairs = build_contrast_pairs("toxicity", mock_toxicity_probes)
        assert len(pairs) == 1  # min(2 benign, 1 toxic) = 1

    def test_sycophancy_pairs(self):
        from rho_eval.interpretability.activation import build_contrast_pairs
        probes = [
            {
                "text": "Is the sky blue?",
                "truthful_answer": "Yes",
                "sycophantic_answer": "No",
                "id": "test",
            }
        ]
        pairs = build_contrast_pairs("sycophancy", probes)
        assert len(pairs) == 1
        assert "Yes" in pairs[0]["positive"]
        assert "No" in pairs[0]["negative"]

    def test_bias_pairs(self):
        from rho_eval.interpretability.activation import build_contrast_pairs
        probes = [
            {
                "text": "Who is the doctor?",
                "correct_answer": "The woman",
                "biased_answer": "The man",
                "id": "test",
            }
        ]
        pairs = build_contrast_pairs("bias", probes)
        assert len(pairs) == 1
        assert "woman" in pairs[0]["positive"]
        assert "man" in pairs[0]["negative"]

    def test_unsupported_behavior(self):
        from rho_eval.interpretability.activation import build_contrast_pairs
        with pytest.raises(ValueError, match="No contrast pair"):
            build_contrast_pairs("reasoning", [])


# ═══════════════════════════════════════════════════════════════════════════
# Import Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestImports:
    """Test that all public API imports work."""

    def test_import_from_interpretability(self):
        from rho_eval.interpretability import (
            extract_subspaces,
            compute_overlap,
            head_attribution,
            orthogonal_project,
            rank_k_steer,
            evaluate_surgical,
            evaluate_baseline,
            SubspaceResult,
            OverlapMatrix,
            HeadImportance,
            SurgicalResult,
            InterpretabilityReport,
        )

    def test_import_from_rho_eval(self):
        from rho_eval import (
            extract_subspaces,
            compute_overlap,
            head_attribution,
            InterpretabilityReport,
        )

    def test_import_visualize(self):
        from rho_eval.interpretability.visualize import (
            plot_overlap_heatmap,
            plot_head_importance,
            plot_dimensionality,
            plot_surgical_comparison,
        )

    def test_import_activation(self):
        from rho_eval.interpretability.activation import (
            LayerActivationCapture,
            HeadOutputCapture,
            SteeringHook,
            build_contrast_pairs,
        )
