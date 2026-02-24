"""Dataset construction for rho-guided SFT.

Two datasets:
  1. SFT dataset: instruction-following data (Alpaca) mixed with
     behavioral trap examples from probe positive texts.
  2. Behavioral contrast dataset: positive/negative text pairs for
     computing the auxiliary contrastive loss during training.

The SFT dataset feeds the standard CE loss. The contrast dataset feeds
the rho auxiliary loss. Together they drive both general capability
and behavioral alignment.
"""

from __future__ import annotations

import random
from typing import Optional

from torch.utils.data import Dataset

from ..calibration import TextDataset
from ..utils import DATA_DIR


CACHE_DIR = DATA_DIR / "alignment_cache"

# Behaviors supported for contrastive loss (not reasoning — GSM8K
# accuracy doesn't translate cleanly to a confidence margin)
CONTRAST_BEHAVIORS = ["factual", "toxicity", "sycophancy", "bias"]


def _load_alpaca_texts(n: int, seed: int = 42) -> list[str]:
    """Load and format Alpaca instruction data.

    Replicates the loading logic from calibration.py load_calibration_data()
    but returns raw text strings instead of a TextDataset.

    Args:
        n: Maximum number of texts to return.
        seed: Random seed for shuffling.

    Returns:
        List of formatted instruction-response strings.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required for SFT data loading. "
            "Install with: pip install datasets"
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    ds = load_dataset(
        "tatsu-lab/alpaca", split="train",
        cache_dir=str(CACHE_DIR),
    )

    texts = []
    for ex in ds:
        if ex.get("input"):
            text = (
                f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Input:\n{ex['input']}\n\n"
                f"### Response:\n{ex['output']}"
            )
        else:
            text = (
                f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Response:\n{ex['output']}"
            )
        texts.append(text)

    rng.shuffle(texts)
    return texts[:n]


def _build_trap_texts(behaviors: list[str], seed: int = 42) -> list[str]:
    """Build trap examples from behavioral probe positive texts.

    Takes the 'positive' text from each contrast pair and formats it
    as an instruction-response example. These examples teach the model
    to be more confident on desired (truthful, non-toxic, non-biased,
    non-sycophantic) text.

    Args:
        behaviors: List of behavior names to include.
        seed: Random seed for probe loading.

    Returns:
        List of instruction-formatted trap texts.
    """
    from ..behaviors import get_behavior
    from ..interpretability.activation import build_contrast_pairs

    texts = []

    for behavior in behaviors:
        try:
            beh = get_behavior(behavior)
            probes = beh.load_probes(seed=seed)
            pairs = build_contrast_pairs(behavior, probes)
        except Exception:
            continue

        for pair in pairs:
            positive = pair["positive"]
            # Format as instruction-response so it fits the SFT training format
            text = (
                f"### Instruction:\n"
                f"Complete the following text accurately and truthfully.\n\n"
                f"### Response:\n{positive}"
            )
            texts.append(text)

    return texts


def load_sft_dataset(
    tokenizer,
    n: int = 2000,
    include_traps: bool = True,
    trap_ratio: float = 0.2,
    behaviors: list[str] | None = None,
    seed: int = 42,
    max_length: int = 256,
) -> TextDataset:
    """Load a mixed SFT dataset with optional behavioral traps.

    Sources:
      - Alpaca instruction data (~80% of budget)
      - Behavioral trap examples from probe positives (~20%)

    The trap examples teach the model to be confident on desired text
    (true facts, benign text, truthful answers, unbiased responses).
    This complements the contrastive loss which explicitly penalizes
    confidence on undesired text.

    Args:
        tokenizer: Tokenizer for encoding (required).
        n: Total number of examples.
        include_traps: Whether to include behavioral trap examples.
        trap_ratio: Fraction of dataset that should be traps (0-1).
        behaviors: Which behaviors to draw traps from.
            Defaults to CONTRAST_BEHAVIORS.
        seed: Random seed.
        max_length: Max token length per example.

    Returns:
        TextDataset ready for causal LM training.
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required for load_sft_dataset")

    rng = random.Random(seed)
    behaviors = behaviors or list(CONTRAST_BEHAVIORS)
    texts = []

    # Source 1: Behavioral trap examples
    if include_traps:
        n_traps = int(n * trap_ratio)
        trap_texts = _build_trap_texts(behaviors, seed=seed)
        rng.shuffle(trap_texts)
        trap_texts = trap_texts[:n_traps]
        texts.extend(trap_texts)
        print(f"  [sft] {len(trap_texts)} behavioral trap texts loaded")

    # Source 2: Alpaca instruction data (fills remaining budget)
    remaining = n - len(texts)
    if remaining > 0:
        try:
            alpaca_texts = _load_alpaca_texts(remaining, seed=seed)
            texts.extend(alpaca_texts)
            print(f"  [sft] {len(alpaca_texts)} Alpaca instruction texts loaded")
        except Exception as e:
            print(f"  [sft] Alpaca load failed ({e}), using traps only")

    # Shuffle combined dataset
    rng.shuffle(texts)
    texts = texts[:n]
    print(f"  [sft] Total: {len(texts)} SFT texts")

    return TextDataset(texts, tokenizer, max_length=max_length)


class BehavioralContrastDataset(Dataset):
    """Dataset of positive/negative text pairs for the auxiliary rho loss.

    Wraps the output of `build_contrast_pairs()` from
    `interpretability/activation.py` for all supported behaviors.
    Each item yields the raw text pair — the training loop tokenizes
    on-the-fly via `rho_auxiliary_loss()`.

    Args:
        behaviors: List of behavior names to include.
            Defaults to CONTRAST_BEHAVIORS.
        seed: Random seed for probe loading.
        max_pairs_per_behavior: Cap on pairs per behavior (None = all).
    """

    def __init__(
        self,
        behaviors: list[str] | None = None,
        seed: int = 42,
        max_pairs_per_behavior: int | None = None,
    ):
        from ..behaviors import get_behavior
        from ..interpretability.activation import build_contrast_pairs

        behaviors = behaviors or list(CONTRAST_BEHAVIORS)
        self.pairs: list[dict] = []

        rng = random.Random(seed)

        for behavior in behaviors:
            try:
                beh = get_behavior(behavior)
                probes = beh.load_probes(seed=seed)
                pairs = build_contrast_pairs(behavior, probes)
            except Exception as e:
                print(f"  [contrast] WARNING: skipping {behavior}: {e}")
                continue

            # Add behavior label to each pair
            for pair in pairs:
                pair["behavior"] = behavior

            if max_pairs_per_behavior and len(pairs) > max_pairs_per_behavior:
                pairs = rng.sample(pairs, max_pairs_per_behavior)

            self.pairs.extend(pairs)

        rng.shuffle(self.pairs)
        print(f"  [contrast] {len(self.pairs)} contrast pairs across "
              f"{len(behaviors)} behaviors")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        """Return a single contrast pair.

        Returns:
            Dict with keys: positive_text, negative_text, behavior.
        """
        pair = self.pairs[idx]
        return {
            "positive_text": pair["positive"],
            "negative_text": pair["negative"],
            "behavior": pair["behavior"],
        }

    def sample(self, k: int, rng: random.Random | None = None) -> list[dict]:
        """Sample k pairs randomly (for batch construction).

        Args:
            k: Number of pairs to sample.
            rng: Optional random.Random instance for reproducibility.

        Returns:
            List of dicts with positive/negative keys (compatible with
            `rho_auxiliary_loss()`).
        """
        rng = rng or random.Random()
        k = min(k, len(self.pairs))
        indices = rng.sample(range(len(self.pairs)), k)
        return [
            {"positive": self.pairs[i]["positive"],
             "negative": self.pairs[i]["negative"]}
            for i in indices
        ]
