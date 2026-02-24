"""Tests for the rho-guided alignment module.

All tests use synthetic data or mock tensors — no model loading required.
The tests verify:
  - Differentiable losses produce tensors with grad_fn
  - Contrastive margin behavior is correct
  - Dataset classes have correct format and length
  - Public API imports work
"""

import pytest
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════
# Helpers — tiny mock model for loss tests
# ═══════════════════════════════════════════════════════════════════════

class _MockOutput:
    """Mimics HuggingFace model output with a loss attribute."""

    def __init__(self, loss):
        self.loss = loss
        self.logits = None


class _MockCausalLM(nn.Module):
    """Tiny mock CausalLM that returns controllable loss values.

    When the input text contains 'positive' or 'true', returns low loss.
    When the input text contains 'negative' or 'false', returns high loss.
    Otherwise returns medium loss.
    """

    def __init__(self):
        super().__init__()
        # Need at least one trainable parameter for grad tests
        self.dummy = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # The loss depends on the first token value as a proxy for content
        # Lower first-token ID → lower loss (more "positive")
        if input_ids is not None:
            first_tok = input_ids[0, 0].float()
            # Scale to a reasonable loss range [0.5, 3.0]
            loss = 0.5 + (first_tok / 100.0) * self.dummy
        else:
            loss = torch.tensor(1.5, requires_grad=True)
        return _MockOutput(loss)

    def generate(self, **kwargs):
        return kwargs.get("input_ids", torch.tensor([[0]]))


class _MockTokenizer:
    """Tiny mock tokenizer."""

    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "</s>"
    eos_token_id = 1
    vocab_size = 100

    def __call__(self, text, return_tensors="pt", truncation=True,
                 max_length=256, padding=None, **kwargs):
        # Encode "positive" texts with low IDs, "negative" with high IDs
        if any(w in text.lower() for w in ["paris", "true", "positive", "benign", "love"]):
            ids = [2, 3, 4, 5]  # Low IDs → low loss
        elif any(w in text.lower() for w in ["berlin", "false", "negative", "toxic", "harmful"]):
            ids = [80, 81, 82, 83]  # High IDs → high loss
        else:
            ids = [40, 41, 42, 43]  # Medium
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens=True):
        return "decoded text"


# ═══════════════════════════════════════════════════════════════════════
# Tests: differentiable_ce_loss
# ═══════════════════════════════════════════════════════════════════════

class TestDifferentiableCELoss:
    def setup_method(self):
        self.model = _MockCausalLM()
        self.tokenizer = _MockTokenizer()

    def test_returns_tensor(self):
        from rho_eval.alignment.losses import differentiable_ce_loss

        loss = differentiable_ce_loss(
            self.model, self.tokenizer, "Paris is the capital", device="cpu",
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar

    def test_has_grad_fn(self):
        from rho_eval.alignment.losses import differentiable_ce_loss

        loss = differentiable_ce_loss(
            self.model, self.tokenizer, "Paris is the capital", device="cpu",
        )
        assert loss.grad_fn is not None

    def test_backward_succeeds(self):
        from rho_eval.alignment.losses import differentiable_ce_loss

        loss = differentiable_ce_loss(
            self.model, self.tokenizer, "Paris is the capital", device="cpu",
        )
        loss.backward()
        assert self.model.dummy.grad is not None

    def test_positive_text_lower_loss(self):
        from rho_eval.alignment.losses import differentiable_ce_loss

        loss_pos = differentiable_ce_loss(
            self.model, self.tokenizer, "Paris is true", device="cpu",
        )
        loss_neg = differentiable_ce_loss(
            self.model, self.tokenizer, "Berlin is false", device="cpu",
        )
        # Positive text (low token IDs) should have lower loss
        assert loss_pos.item() < loss_neg.item()


# ═══════════════════════════════════════════════════════════════════════
# Tests: contrastive_confidence_loss
# ═══════════════════════════════════════════════════════════════════════

class TestContrastiveConfidenceLoss:
    def setup_method(self):
        self.model = _MockCausalLM()
        self.tokenizer = _MockTokenizer()

    def test_zero_when_well_separated(self):
        from rho_eval.alignment.losses import contrastive_confidence_loss

        # Positive text has much lower loss than negative
        loss = contrastive_confidence_loss(
            self.model, self.tokenizer,
            "Paris is true positive", "Berlin is false negative",
            margin=0.01, device="cpu",
        )
        # Should be 0 (or very close) since positive is already more confident
        assert loss.item() < 0.01

    def test_positive_when_reversed(self):
        from rho_eval.alignment.losses import contrastive_confidence_loss

        # Swap: "negative" as positive, "positive" as negative
        loss = contrastive_confidence_loss(
            self.model, self.tokenizer,
            "Berlin is false negative",  # "positive" arg but high loss
            "Paris is true positive",    # "negative" arg but low loss
            margin=0.1, device="cpu",
        )
        # Should be > 0 since "positive" arg has higher loss
        assert loss.item() > 0

    def test_margin_increases_loss(self):
        from rho_eval.alignment.losses import contrastive_confidence_loss

        loss_small = contrastive_confidence_loss(
            self.model, self.tokenizer,
            "Something neutral text", "Berlin is false negative",
            margin=0.01, device="cpu",
        )
        loss_large = contrastive_confidence_loss(
            self.model, self.tokenizer,
            "Something neutral text", "Berlin is false negative",
            margin=1.0, device="cpu",
        )
        # Larger margin should produce equal or larger loss
        assert loss_large.item() >= loss_small.item()

    def test_gradient_flows(self):
        from rho_eval.alignment.losses import contrastive_confidence_loss

        loss = contrastive_confidence_loss(
            self.model, self.tokenizer,
            "Berlin is false negative",  # reversed to ensure non-zero loss
            "Paris is true positive",
            margin=0.1, device="cpu",
        )
        loss.backward()
        assert self.model.dummy.grad is not None


# ═══════════════════════════════════════════════════════════════════════
# Tests: rho_auxiliary_loss
# ═══════════════════════════════════════════════════════════════════════

class TestRhoAuxiliaryLoss:
    def setup_method(self):
        self.model = _MockCausalLM()
        self.tokenizer = _MockTokenizer()

    def test_empty_pairs_returns_zero(self):
        from rho_eval.alignment.losses import rho_auxiliary_loss

        loss = rho_auxiliary_loss(self.model, self.tokenizer, [], device="cpu")
        assert loss.item() == 0.0

    def test_batch_produces_scalar(self):
        from rho_eval.alignment.losses import rho_auxiliary_loss

        pairs = [
            {"positive": "Paris is true", "negative": "Berlin is false"},
            {"positive": "Love and benign", "negative": "Harmful toxic text"},
        ]
        loss = rho_auxiliary_loss(self.model, self.tokenizer, pairs, device="cpu")
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_differentiable(self):
        from rho_eval.alignment.losses import rho_auxiliary_loss

        pairs = [
            {"positive": "Berlin false negative",  # reversed
             "negative": "Paris true positive"},
        ]
        loss = rho_auxiliary_loss(self.model, self.tokenizer, pairs, device="cpu")
        loss.backward()
        assert self.model.dummy.grad is not None

    def test_uses_mock_contrast_pairs(self, mock_contrast_pairs):
        from rho_eval.alignment.losses import rho_auxiliary_loss

        # mock_contrast_pairs uses 'positive'/'negative' keys — compatible
        loss = rho_auxiliary_loss(
            self.model, self.tokenizer, mock_contrast_pairs, device="cpu",
        )
        assert isinstance(loss, torch.Tensor)


# ═══════════════════════════════════════════════════════════════════════
# Tests: BehavioralContrastDataset
# ═══════════════════════════════════════════════════════════════════════

class TestBehavioralContrastDataset:
    def test_loads_factual_pairs(self):
        from rho_eval.alignment.dataset import BehavioralContrastDataset

        ds = BehavioralContrastDataset(behaviors=["factual"], seed=42)
        assert len(ds) > 0

    def test_item_format(self):
        from rho_eval.alignment.dataset import BehavioralContrastDataset

        ds = BehavioralContrastDataset(behaviors=["factual"], seed=42)
        item = ds[0]
        assert "positive_text" in item
        assert "negative_text" in item
        assert "behavior" in item
        assert item["behavior"] == "factual"

    def test_sample_returns_compatible_format(self):
        from rho_eval.alignment.dataset import BehavioralContrastDataset

        ds = BehavioralContrastDataset(behaviors=["factual"], seed=42)
        sampled = ds.sample(3)
        assert len(sampled) <= 3
        for pair in sampled:
            assert "positive" in pair
            assert "negative" in pair

    def test_max_pairs_per_behavior(self):
        from rho_eval.alignment.dataset import BehavioralContrastDataset

        ds_full = BehavioralContrastDataset(behaviors=["factual"], seed=42)
        ds_capped = BehavioralContrastDataset(
            behaviors=["factual"], seed=42, max_pairs_per_behavior=5,
        )
        assert len(ds_capped) <= 5
        assert len(ds_capped) <= len(ds_full)

    def test_multiple_behaviors(self):
        from rho_eval.alignment.dataset import BehavioralContrastDataset

        ds = BehavioralContrastDataset(
            behaviors=["factual", "toxicity"], seed=42,
        )
        behaviors_found = set(ds[i]["behavior"] for i in range(len(ds)))
        # Should have at least factual (toxicity may fail if HF dataset not cached)
        assert "factual" in behaviors_found


# ═══════════════════════════════════════════════════════════════════════
# Tests: Imports
# ═══════════════════════════════════════════════════════════════════════

class TestImports:
    def test_import_from_alignment(self):
        from rho_eval.alignment import (
            differentiable_ce_loss,
            contrastive_confidence_loss,
            rho_auxiliary_loss,
            load_sft_dataset,
            BehavioralContrastDataset,
            rho_guided_sft,
        )
        assert callable(differentiable_ce_loss)
        assert callable(contrastive_confidence_loss)
        assert callable(rho_auxiliary_loss)
        assert callable(load_sft_dataset)
        assert callable(rho_guided_sft)

    def test_import_from_rho_eval(self):
        from rho_eval import (
            rho_guided_sft,
            contrastive_confidence_loss,
            rho_auxiliary_loss,
        )
        assert callable(rho_guided_sft)
        assert callable(contrastive_confidence_loss)
        assert callable(rho_auxiliary_loss)

    def test_import_losses_module(self):
        from rho_eval.alignment.losses import (
            differentiable_ce_loss,
            contrastive_confidence_loss,
            rho_auxiliary_loss,
        )
        assert callable(differentiable_ce_loss)

    def test_import_dataset_module(self):
        from rho_eval.alignment.dataset import (
            load_sft_dataset,
            BehavioralContrastDataset,
            CONTRAST_BEHAVIORS,
        )
        assert isinstance(CONTRAST_BEHAVIORS, list)
        assert "factual" in CONTRAST_BEHAVIORS

    def test_import_trainer_module(self):
        from rho_eval.alignment.trainer import rho_guided_sft
        assert callable(rho_guided_sft)
