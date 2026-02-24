"""Reasoning behavior — accuracy under adversarial flattery.

Uses generation on GSM8K problems prefixed with flattery to test whether
a model's reasoning accuracy degrades when given praise/flattery.

rho = accuracy on flattery-prefixed GSM8K problems.
"""

from __future__ import annotations

import time
from typing import Optional

from . import register
from .base import ABCBehavior, BehaviorResult
from .metrics import generate, check_numeric


@register
class ReasoningBehavior(ABCBehavior):
    name = "reasoning"
    description = "Reasoning accuracy under adversarial flattery (GSM8K)"
    probe_type = "generation"
    default_n = 100

    def load_probes(self, n: Optional[int] = None, seed: int = 42, **kwargs) -> list[dict]:
        """Load pre-sampled GSM8K + flattery probes."""
        n = n or self.default_n
        return self._load_json_probes("reasoning/gsm8k_100.json", n=n, seed=seed)

    def evaluate(
        self,
        model,
        tokenizer,
        probes: list[dict],
        device: str = "cpu",
        **kwargs,
    ) -> BehaviorResult:
        """Evaluate reasoning under adversarial flattery.

        rho = accuracy on flattery-prefixed GSM8K problems.
        Also computes clean accuracy for comparison.
        """
        t0 = time.time()
        correct_adv = 0
        correct_clean = 0
        details = []

        for probe in probes:
            gen_adv = generate(
                model, tokenizer, probe["text"],
                max_new_tokens=100, device=device,
            )
            gen_clean = generate(
                model, tokenizer, probe["text_clean"],
                max_new_tokens=100, device=device,
            )

            adv_ok = check_numeric(gen_adv, probe["target_answer"])
            clean_ok = check_numeric(gen_clean, probe["target_answer"])

            if adv_ok:
                correct_adv += 1
            if clean_ok:
                correct_clean += 1

            details.append({
                "id": probe["id"],
                "target": probe["target_answer"],
                "adversarial_correct": adv_ok,
                "clean_correct": clean_ok,
            })

        n_probes = len(probes)
        elapsed = time.time() - t0
        adv_acc = correct_adv / n_probes if n_probes > 0 else 0.0
        clean_acc = correct_clean / n_probes if n_probes > 0 else 0.0

        return BehaviorResult(
            behavior=self.name,
            rho=adv_acc,
            retention=adv_acc / clean_acc if clean_acc > 0 else 1.0,
            positive_count=correct_adv,
            total=n_probes,
            elapsed=elapsed,
            metadata={
                "adversarial_accuracy": adv_acc,
                "clean_accuracy": clean_acc,
                "accuracy_drop": clean_acc - adv_acc,
            },
            details=details,
        )
