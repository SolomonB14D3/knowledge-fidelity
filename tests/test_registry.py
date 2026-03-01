"""Tests for the behavior registry and plugin system."""

import pytest


class TestBehaviorRegistry:
    """Test behavior registration, discovery, and instantiation."""

    def test_list_behaviors_returns_all(self):
        from rho_eval.behaviors import list_behaviors
        behaviors = list_behaviors()
        assert len(behaviors) == 8
        for expected in ["bias", "deception", "factual", "overrefusal",
                         "reasoning", "refusal", "sycophancy", "toxicity"]:
            assert expected in behaviors

    def test_get_behavior_returns_instance(self):
        from rho_eval.behaviors import get_behavior
        from rho_eval.behaviors.base import ABCBehavior

        for name in ["factual", "toxicity", "bias", "sycophancy", "reasoning",
                      "refusal", "deception", "overrefusal"]:
            b = get_behavior(name)
            assert isinstance(b, ABCBehavior)
            assert b.name == name

    def test_get_behavior_unknown_raises(self):
        from rho_eval.behaviors import get_behavior

        with pytest.raises(ValueError, match="Unknown behavior"):
            get_behavior("nonexistent")

    def test_get_all_behaviors(self):
        from rho_eval.behaviors import get_all_behaviors
        from rho_eval.behaviors.base import ABCBehavior

        all_b = get_all_behaviors()
        assert len(all_b) == 8
        for name, b in all_b.items():
            assert isinstance(b, ABCBehavior)
            assert b.name == name

    def test_behavior_attributes(self):
        from rho_eval.behaviors import get_behavior

        b = get_behavior("factual")
        assert b.name == "factual"
        assert b.probe_type == "confidence"
        assert b.default_n >= 56  # expanded with bridge probes
        assert len(b.description) > 0

        b = get_behavior("toxicity")
        assert b.probe_type == "confidence"
        assert b.default_n >= 200

        b = get_behavior("bias")
        assert b.probe_type == "generation"
        assert b.default_n >= 300

    def test_behavior_load_probes(self):
        from rho_eval.behaviors import get_behavior

        for name in ["factual", "toxicity", "bias", "sycophancy", "reasoning",
                      "refusal", "deception", "overrefusal"]:
            b = get_behavior(name)
            probes = b.load_probes()
            # Bias now includes bridge probes by default (> default_n)
            assert len(probes) >= b.default_n
            assert all(isinstance(p, dict) for p in probes)

    def test_behavior_load_probes_subsample(self):
        from rho_eval.behaviors import get_behavior

        b = get_behavior("bias")
        probes = b.load_probes(n=10, seed=42)
        assert len(probes) == 10

    def test_behavior_repr(self):
        from rho_eval.behaviors import get_behavior

        b = get_behavior("factual")
        r = repr(b)
        assert "factual" in r
        assert str(b.default_n) in r


class TestCustomBehavior:
    """Test registering a custom behavior."""

    def test_register_custom_behavior(self):
        from rho_eval.behaviors import register, get_behavior, list_behaviors, _REGISTRY
        from rho_eval.behaviors.base import ABCBehavior, BehaviorResult

        # Create and register a custom behavior
        @register
        class TestBehavior(ABCBehavior):
            name = "test_custom"
            description = "A test behavior"
            probe_type = "confidence"
            default_n = 5

            def load_probes(self, n=None, seed=42, **kwargs):
                return [{"text": f"probe_{i}", "id": f"test_{i}"} for i in range(n or self.default_n)]

            def evaluate(self, model, tokenizer, probes, device="cpu", **kwargs):
                return BehaviorResult(
                    behavior=self.name, rho=0.5, retention=0.5,
                    positive_count=len(probes) // 2, total=len(probes),
                )

        # Verify it's registered
        assert "test_custom" in list_behaviors()
        b = get_behavior("test_custom")
        assert b.name == "test_custom"
        probes = b.load_probes()
        assert len(probes) == 5

        # Clean up
        del _REGISTRY["test_custom"]
