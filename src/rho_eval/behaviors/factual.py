"""Factual confidence behavior — true/false probe discrimination.

Uses teacher-forced confidence (via the cartography engine) to measure
whether a model assigns higher confidence to true statements than false ones.

rho = Spearman correlation between true and false confidence vectors.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from . import register
from .base import ABCBehavior, BehaviorResult


@register
class FactualBehavior(ABCBehavior):
    name = "factual"
    description = "True/false statement discrimination via teacher-forced confidence"
    probe_type = "confidence"
    default_n = 56  # 20 default + 6 mandela + 5 medical + 10 commonsense + 15 truthfulqa

    # JSON files that make up the factual probe set
    _PROBE_FILES = [
        "factual/default.json",
        "factual/mandela.json",
        "factual/medical.json",
        "factual/commonsense.json",
        "factual/truthfulqa.json",
    ]

    def load_probes(self, n: Optional[int] = None, seed: int = 42, **kwargs) -> list[dict]:
        """Load factual probes from all shipped JSON files.

        Concatenates default, mandela, medical, commonsense, and truthfulqa
        probe sets. Subsamples if n < total available.
        """
        import random

        all_probes = []
        for filename in self._PROBE_FILES:
            try:
                all_probes.extend(self._load_json_probes(filename))
            except FileNotFoundError:
                pass  # gracefully skip missing optional files

        if not all_probes:
            raise FileNotFoundError(
                "No factual probe data found. "
                "Run `python scripts/presample_probes.py` to generate probe data."
            )

        n = n or self.default_n
        if n < len(all_probes):
            rng = random.Random(seed)
            all_probes = rng.sample(all_probes, n)

        return all_probes

    def evaluate(
        self,
        model,
        tokenizer,
        probes: list[dict],
        device: str = "cpu",
        **kwargs,
    ) -> BehaviorResult:
        """Evaluate factual probes using the core confidence system.

        Each probe must have "text" (true) and "false" keys.
        rho = Spearman correlation between true and false confidence.
        """
        from scipy import stats as sp_stats
        from ..cartography.engine import analyze_confidence

        t0 = time.time()
        true_confs = []
        false_confs = []
        details = []

        for probe in probes:
            try:
                rec_true = analyze_confidence(
                    text=probe["text"],
                    category="true",
                    label=probe.get("id", ""),
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                )
                rec_false = analyze_confidence(
                    text=probe["false"],
                    category="false",
                    label=probe.get("id", ""),
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                )
                true_confs.append(rec_true.mean_top1_prob)
                false_confs.append(rec_false.mean_top1_prob)
                details.append({
                    "id": probe.get("id", ""),
                    "true_conf": rec_true.mean_top1_prob,
                    "false_conf": rec_false.mean_top1_prob,
                    "delta": rec_true.mean_top1_prob - rec_false.mean_top1_prob,
                })
            except Exception as e:
                true_confs.append(0.0)
                false_confs.append(0.0)
                details.append({"id": probe.get("id", ""), "error": str(e)})

        deltas = np.array(true_confs) - np.array(false_confs)
        n_positive = int((deltas > 0).sum())

        if len(true_confs) >= 3:
            rho, _ = sp_stats.spearmanr(true_confs, false_confs)
        else:
            rho = 0.0

        elapsed = time.time() - t0

        return BehaviorResult(
            behavior=self.name,
            rho=float(rho),
            retention=n_positive / len(probes) if probes else 0.0,
            positive_count=n_positive,
            total=len(probes),
            elapsed=elapsed,
            metadata={"mean_delta": float(deltas.mean())},
            details=details,
        )
