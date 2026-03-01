"""Tests for the probe system — counts, loading, validation."""

import json
import tempfile
from pathlib import Path

import pytest


class TestProbeRegistry:
    """Test the probe discovery and loading system."""

    def test_list_probe_sets_returns_all(self):
        from rho_eval.probes import list_probe_sets
        sets = list_probe_sets()
        assert len(sets) >= 12  # at least 9 original + 3 bench (more with bridge probes)
        assert "factual/default" in sets
        assert "bias/bbq_300" in sets
        assert "toxicity/toxigen_200" in sets
        assert "sycophancy/anthropic_150" in sets
        assert "reasoning/gsm8k_100" in sets
        assert "bench/logic" in sets
        assert "bench/social" in sets
        assert "bench/clinical" in sets

    def test_probe_counts_match_paper(self):
        """Probe counts must match the claims in the paper."""
        from rho_eval.probes import get_probe_counts

        counts = get_probe_counts()

        # Factual probes
        assert counts["factual/default"] == 20
        assert counts["factual/mandela"] == 6
        assert counts["factual/medical"] == 5
        assert counts["factual/commonsense"] == 10
        assert counts["factual/truthfulqa"] == 15

        # Other behaviors
        assert counts["bias/bbq_300"] == 300
        assert counts["sycophancy/anthropic_150"] == 150
        assert counts["toxicity/toxigen_200"] == 200
        assert counts["reasoning/gsm8k_100"] == 100

        # Bench probes
        assert counts["bench/logic"] == 40
        assert counts["bench/social"] == 40
        assert counts["bench/clinical"] == 40

        # Total: at least 926 (original + bench), more with bridge probes
        total = sum(counts.values())
        assert total >= 926

    def test_get_probes_loads_valid_data(self):
        from rho_eval.probes import get_probes

        probes = get_probes("factual/default")
        assert len(probes) == 20
        for p in probes:
            assert "text" in p
            assert "false" in p
            assert "id" in p

    def test_get_probes_subsample(self):
        from rho_eval.probes import get_probes

        probes = get_probes("bias/bbq_300", n=10, seed=42)
        assert len(probes) == 10

    def test_get_probes_unknown_raises(self):
        from rho_eval.probes import get_probes

        with pytest.raises(FileNotFoundError, match="not found"):
            get_probes("nonexistent/set")


class TestProbeValidation:
    """Test that all shipped probes have required fields."""

    def test_factual_probes_have_false_key(self):
        from rho_eval.probes import get_probes

        for name in ["factual/default", "factual/mandela", "factual/medical"]:
            probes = get_probes(name)
            for p in probes:
                assert "false" in p, f"Probe {p.get('id', '?')} in {name} missing 'false'"

    def test_toxicity_probes_have_is_toxic(self):
        from rho_eval.probes import get_probes

        probes = get_probes("toxicity/toxigen_200")
        for p in probes:
            assert "is_toxic" in p
            assert isinstance(p["is_toxic"], bool)

    def test_bias_probes_have_correct_answer(self):
        from rho_eval.probes import get_probes

        probes = get_probes("bias/bbq_300")
        for p in probes:
            assert "correct_answer" in p
            assert p["correct_answer"] in ("A", "B", "C")

    def test_sycophancy_probes_have_answers(self):
        from rho_eval.probes import get_probes

        probes = get_probes("sycophancy/anthropic_150")
        for p in probes:
            assert "truthful_answer" in p
            assert "sycophantic_answer" in p

    def test_reasoning_probes_have_target(self):
        from rho_eval.probes import get_probes

        probes = get_probes("reasoning/gsm8k_100")
        for p in probes:
            assert "target_answer" in p
            assert "text_clean" in p


class TestLegacyProbeAPI:
    """Backward compatibility with the old probes.py module."""

    def test_get_default_probes(self):
        from rho_eval.probes import get_default_probes
        probes = get_default_probes()
        assert len(probes) == 20

    def test_get_mandela_probes(self):
        from rho_eval.probes import get_mandela_probes
        probes = get_mandela_probes()
        assert len(probes) == 6

    def test_get_medical_probes(self):
        from rho_eval.probes import get_medical_probes
        probes = get_medical_probes()
        assert len(probes) == 5

    def test_get_importance_prompts(self):
        from rho_eval.probes import get_importance_prompts
        prompts = get_importance_prompts()
        assert len(prompts) == 20
        assert all(isinstance(p, str) for p in prompts)

    def test_save_and_load_probes(self):
        from rho_eval.probes import save_probes, load_probes

        probes = [
            {"text": "Test true.", "false": "Test false.", "domain": "test", "id": "test_0"},
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        save_probes(probes, path)
        loaded = load_probes(path)
        assert len(loaded) == 1
        assert loaded[0]["text"] == "Test true."
        Path(path).unlink()
