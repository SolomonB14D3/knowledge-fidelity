#!/usr/bin/env python3
"""Phase 3: Extract and evaluate behavioral steering vectors.

Uses Contrastive Activation Addition (CAA) to extract steering vectors from
rho-audit behavioral probes, then evaluates whether they can steer model
behavior at inference time.

Method (Rimsky et al., 2023 — "Steering Llama 2 via Contrastive Activation Addition"):
  1. Construct contrast pairs from behavioral probes (positive vs negative examples)
  2. Run forward passes, hooking into residual stream at each layer
  3. Compute mean activation difference → steering vector per layer per behavior
  4. Sweep layer + alpha to find best steering configuration
  5. Evaluate: does applying the vector change rho scores?

This connects the existing rho-audit infrastructure with activation engineering:
  - Probes provide the contrast pairs (already validated in Phase 2)
  - Rho provides the evaluation metric (already calibrated across architectures)
  - Layer localization (Phase 1) tells us where to look

Designed as an overnight experiment (~4-8 hours on M3 Ultra for full sweep).

Usage:
    python experiments/steering_vectors.py
    python experiments/steering_vectors.py --model Qwen/Qwen2.5-7B-Instruct
    python experiments/steering_vectors.py --behaviors sycophancy --quick
"""

import argparse
import gc
import json
import math
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from knowledge_fidelity.behavioral import (
    load_behavioral_probes,
    evaluate_behavior,
)
from knowledge_fidelity.probes import get_all_probes
from knowledge_fidelity.utils import get_layers, free_memory

RESULTS_DIR = Path(__file__).parent.parent / "results" / "steering"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ── Activation Extraction ─────────────────────────────────────────────────

class ActivationCapture:
    """Hook-based activation capture from residual stream.

    Captures the output hidden states from a transformer layer during
    forward passes. Used to build contrast pair activation datasets
    for steering vector extraction.

    Usage:
        cap = ActivationCapture(model, layer_idx=15)
        model(**inputs)  # forward pass
        activations = cap.get()  # (batch, seq_len, hidden_dim)
        cap.remove()  # cleanup
    """

    def __init__(self, model, layer_idx: int):
        self.activations = None
        layers = get_layers(model)
        self._hook = layers[layer_idx].register_forward_hook(self._capture)

    def _capture(self, module, input, output):
        # Transformer layers output (hidden_states, ...) tuple
        if isinstance(output, tuple):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()

    def get(self) -> torch.Tensor:
        """Return captured activations. Shape: (batch, seq_len, hidden_dim)."""
        return self.activations

    def remove(self):
        """Remove the hook."""
        self._hook.remove()


class SteeringHook:
    """Apply a steering vector during forward pass.

    Adds a scaled vector to the residual stream at a specific layer.
    Applied at all token positions (CAA-style).

    Usage:
        hook = SteeringHook(model, layer_idx=15, vector=v, alpha=2.0)
        model.generate(...)  # steered generation
        hook.remove()
    """

    def __init__(self, model, layer_idx: int, vector: torch.Tensor, alpha: float = 1.0):
        self.vector = vector
        self.alpha = alpha
        layers = get_layers(model)
        self._hook = layers[layer_idx].register_forward_hook(self._steer)

    def _steer(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            # Add steering vector to all positions: (batch, seq, hidden) + (hidden,)
            h = h + self.alpha * self.vector.to(h.device, h.dtype)
            return (h,) + output[1:]
        else:
            return output + self.alpha * self.vector.to(output.device, output.dtype)

    def remove(self):
        self._hook.remove()


# ── Contrast Pair Construction ────────────────────────────────────────────

def build_contrast_pairs(behavior: str, probes: list[dict]) -> list[dict]:
    """Build contrast pairs from behavioral probes.

    Each pair has a 'positive' text (desired behavior) and 'negative' text
    (undesired behavior). The specific construction depends on the behavior.

    Returns:
        List of dicts with keys: positive, negative, id
    """
    pairs = []

    if behavior == "factual":
        # True statement (positive) vs false statement (negative)
        for p in probes:
            if "text" in p and "false" in p:
                pairs.append({
                    "positive": p["text"],
                    "negative": p["false"],
                    "id": p.get("id", ""),
                })

    elif behavior == "sycophancy":
        # Truthful answer (positive) vs sycophantic answer (negative)
        # We use the full prompt + correct answer vs prompt + sycophantic answer
        for p in probes:
            text = p["text"]
            pairs.append({
                "positive": f"{text}\n{p['truthful_answer']}",
                "negative": f"{text}\n{p['sycophantic_answer']}",
                "id": p.get("id", ""),
            })

    elif behavior == "bias":
        # Construct: prompt + correct answer vs prompt + biased answer
        for p in probes:
            text = p["text"]
            pairs.append({
                "positive": f"{text} {p['correct_answer']}",
                "negative": f"{text} {p['biased_answer']}",
                "id": p.get("id", ""),
            })

    elif behavior == "toxicity":
        # Benign text (positive) vs toxic text (negative)
        benign = [p for p in probes if not p.get("is_toxic", False)]
        toxic = [p for p in probes if p.get("is_toxic", False)]
        n = min(len(benign), len(toxic))
        for i in range(n):
            pairs.append({
                "positive": benign[i]["text"],
                "negative": toxic[i]["text"],
                "id": f"tox_pair_{i}",
            })

    else:
        raise ValueError(f"No contrast pair construction for behavior: {behavior}")

    return pairs


# ── Steering Vector Extraction ────────────────────────────────────────────

@torch.no_grad()
def extract_steering_vectors(
    model,
    tokenizer,
    pairs: list[dict],
    layers: list[int],
    device: str = "cpu",
    method: str = "mean_diff",  # "mean_diff" or "pca"
) -> dict[int, torch.Tensor]:
    """Extract steering vectors from contrast pairs at specified layers.

    For each layer, runs forward passes on all positive and negative texts,
    extracts last-token activations, and computes the mean difference.

    Args:
        model: HuggingFace causal LM
        tokenizer: corresponding tokenizer
        pairs: list of dicts with 'positive' and 'negative' keys
        layers: list of layer indices to extract from
        device: torch device string
        method: "mean_diff" (CAA-style) or "pca" (RepE-style)

    Returns:
        Dict mapping layer_idx -> steering_vector (shape: hidden_dim)
    """
    print(f"  Extracting steering vectors from {len(pairs)} pairs "
          f"at {len(layers)} layers ({method})...")

    vectors = {}

    for layer_idx in layers:
        pos_activations = []
        neg_activations = []

        cap = ActivationCapture(model, layer_idx)

        for pair in pairs:
            # Positive
            inputs = tokenizer(
                pair["positive"], return_tensors="pt",
                truncation=True, max_length=512,
            ).to(device)
            model(**inputs)
            h_pos = cap.get()  # (1, seq_len, hidden_dim)
            pos_activations.append(h_pos[0, -1, :].cpu())  # last token

            # Negative
            inputs = tokenizer(
                pair["negative"], return_tensors="pt",
                truncation=True, max_length=512,
            ).to(device)
            model(**inputs)
            h_neg = cap.get()
            neg_activations.append(h_neg[0, -1, :].cpu())

        cap.remove()

        # Stack: (n_pairs, hidden_dim)
        pos_stack = torch.stack(pos_activations)
        neg_stack = torch.stack(neg_activations)

        if method == "mean_diff":
            # CAA: simple mean difference
            vector = pos_stack.mean(dim=0) - neg_stack.mean(dim=0)
        elif method == "pca":
            # RepE: PCA on difference vectors
            diffs = pos_stack - neg_stack  # (n_pairs, hidden_dim)
            # Center
            diffs_centered = diffs - diffs.mean(dim=0)
            # SVD → first principal component
            U, S, Vh = torch.linalg.svd(diffs_centered, full_matrices=False)
            vector = Vh[0]  # first right singular vector
            # Sign convention: align with mean difference direction
            if torch.dot(vector, (pos_stack.mean(0) - neg_stack.mean(0))) < 0:
                vector = -vector
        else:
            raise ValueError(f"Unknown method: {method}")

        vectors[layer_idx] = vector
        print(f"    Layer {layer_idx}: ||v|| = {vector.norm():.4f}")

    return vectors


# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate_steered(
    model,
    tokenizer,
    behavior: str,
    probes: list[dict],
    vector: torch.Tensor,
    layer_idx: int,
    alpha: float,
    device: str = "cpu",
) -> dict:
    """Evaluate a model with a steering vector applied.

    Hooks the steering vector into the specified layer during evaluation,
    then runs the standard behavioral evaluation.

    Returns:
        Dict with rho and other behavior-specific metrics.
    """
    hook = SteeringHook(model, layer_idx, vector, alpha)
    try:
        result = evaluate_behavior(behavior, model, tokenizer, probes, device)
    finally:
        hook.remove()
    return result


# ── Main Experiment ───────────────────────────────────────────────────────

def load_model(model_id, device=DEVICE):
    """Load model + tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading {model_id}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32, trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded in {time.time()-t0:.1f}s ({n_params/1e9:.2f}B params)")
    return model, tokenizer


def _save(results, path):
    """Atomic JSON save with NaN handling."""
    def _default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if math.isnan(v) else v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return float(obj)

    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=_default)
    tmp.rename(path)


def run_steering_experiment(
    model_id: str,
    behaviors: list[str],
    device: str = DEVICE,
    method: str = "mean_diff",
    alpha_range: list[float] = None,
    layer_pcts: list[float] = None,
    quick: bool = False,
):
    """Full steering vector extraction and evaluation experiment.

    Steps:
    1. Load model
    2. Compute baseline rho for each behavior
    3. Build contrast pairs from probes
    4. Extract steering vectors at candidate layers
    5. Sweep alpha values, evaluating steered rho
    6. Save results

    Args:
        model_id: HuggingFace model name
        behaviors: list of behaviors to test
        device: torch device
        method: "mean_diff" or "pca"
        alpha_range: list of alpha multipliers to sweep
        layer_pcts: list of layer depth percentages to test (0.0-1.0)
        quick: if True, use fewer alphas and layers
    """
    if alpha_range is None:
        if quick:
            alpha_range = [-2.0, -1.0, 0.5, 1.0, 2.0]
        else:
            alpha_range = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0]

    if layer_pcts is None:
        if quick:
            layer_pcts = [0.25, 0.50, 0.75]
        else:
            layer_pcts = [0.25, 0.375, 0.50, 0.625, 0.75, 0.875]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_short = model_id.split("/")[-1].lower().replace("-", "_")
    output_path = RESULTS_DIR / f"steering_{model_short}.json"

    # Resume if exists
    results = {}
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        print(f"Resumed {len(results.get('behaviors', {}))} behaviors from {output_path}")

    model, tokenizer = load_model(model_id, device)
    n_layers = len(get_layers(model))
    candidate_layers = sorted(set(
        max(0, min(n_layers - 1, int(pct * n_layers)))
        for pct in layer_pcts
    ))

    print(f"\nModel: {model_id} ({n_layers} layers)")
    print(f"Candidate layers: {candidate_layers}")
    print(f"Alpha range: {alpha_range}")
    print(f"Method: {method}")
    print(f"Behaviors: {behaviors}")

    results["model_id"] = model_id
    results["n_layers"] = n_layers
    results["method"] = method
    results["candidate_layers"] = candidate_layers
    results["alpha_range"] = alpha_range
    results["device"] = str(device)
    results["timestamp_start"] = datetime.now().isoformat()
    results.setdefault("behaviors", {})

    for behavior in behaviors:
        if behavior in results["behaviors"]:
            print(f"\n  [{behavior}] Already done — skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  BEHAVIOR: {behavior}")
        print(f"{'='*60}")

        bt0 = time.time()

        # ── Load probes ──
        if behavior == "factual":
            probes = get_all_probes()
        else:
            probes = load_behavioral_probes(behavior, seed=42)
        print(f"  Loaded {len(probes)} probes")

        # ── Baseline evaluation ──
        print(f"  Computing baseline rho...")
        baseline = evaluate_behavior(behavior, model, tokenizer, probes, device)
        baseline_rho = baseline["rho"]
        baseline_clean = {k: v for k, v in baseline.items() if k != "details"}
        print(f"  Baseline rho = {baseline_rho:.4f}")

        # ── Build contrast pairs ──
        pairs = build_contrast_pairs(behavior, probes)
        print(f"  Built {len(pairs)} contrast pairs")

        if len(pairs) < 5:
            print(f"  WARNING: Only {len(pairs)} pairs — skipping (need >= 5)")
            results["behaviors"][behavior] = {
                "baseline": baseline_clean,
                "error": f"Too few contrast pairs ({len(pairs)})",
            }
            _save(results, output_path)
            continue

        # ── Extract steering vectors ──
        vectors = extract_steering_vectors(
            model, tokenizer, pairs, candidate_layers, device, method
        )

        # ── Sweep: layer x alpha ──
        print(f"\n  Sweeping {len(candidate_layers)} layers x {len(alpha_range)} alphas...")
        sweep_results = []

        for layer_idx in candidate_layers:
            vector = vectors[layer_idx]

            for alpha in alpha_range:
                print(f"    Layer {layer_idx}, alpha={alpha:+.1f}...", end=" ", flush=True)
                st0 = time.time()

                steered = evaluate_steered(
                    model, tokenizer, behavior, probes,
                    vector, layer_idx, alpha, device,
                )
                steered_rho = steered["rho"]
                delta = steered_rho - baseline_rho if (
                    steered_rho is not None and baseline_rho is not None
                    and not (isinstance(steered_rho, float) and math.isnan(steered_rho))
                    and not (isinstance(baseline_rho, float) and math.isnan(baseline_rho))
                ) else None

                steered_clean = {k: v for k, v in steered.items() if k != "details"}
                entry = {
                    "layer": layer_idx,
                    "layer_pct": round(layer_idx / n_layers, 3),
                    "alpha": alpha,
                    "rho": steered_rho,
                    "delta_rho": delta,
                    "elapsed_s": round(time.time() - st0, 1),
                    "metrics": steered_clean,
                }
                sweep_results.append(entry)

                delta_str = f"delta={delta:+.4f}" if delta is not None else "delta=N/A"
                print(f"rho={steered_rho:.4f} ({delta_str}) [{time.time()-st0:.0f}s]")

        # ── Find best configuration ──
        valid = [r for r in sweep_results if r["delta_rho"] is not None]
        if valid:
            best = max(valid, key=lambda r: r["delta_rho"])
            worst = min(valid, key=lambda r: r["delta_rho"])
        else:
            best = worst = None

        # ── Store vector norms for reference ──
        vector_norms = {
            str(layer_idx): round(float(vectors[layer_idx].norm()), 4)
            for layer_idx in candidate_layers
        }

        behavior_result = {
            "baseline": baseline_clean,
            "n_pairs": len(pairs),
            "sweep": sweep_results,
            "vector_norms": vector_norms,
            "best": best,
            "worst": worst,
            "elapsed_seconds": round(time.time() - bt0, 1),
        }
        results["behaviors"][behavior] = behavior_result
        _save(results, output_path)

        if best:
            print(f"\n  BEST:  layer={best['layer']}, alpha={best['alpha']:+.1f} "
                  f"→ rho={best['rho']:.4f} (delta={best['delta_rho']:+.4f})")
        if worst:
            print(f"  WORST: layer={worst['layer']}, alpha={worst['alpha']:+.1f} "
                  f"→ rho={worst['rho']:.4f} (delta={worst['delta_rho']:+.4f})")
        print(f"  Elapsed: {time.time()-bt0:.0f}s")

    # ── Save steering vectors to disk ──
    # (Re-extract for the best configs to save the actual vectors)
    results["timestamp_end"] = datetime.now().isoformat()
    _save(results, output_path)

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"  STEERING VECTOR SUMMARY: {model_id}")
    print(f"{'='*60}")
    for bname, bdata in results["behaviors"].items():
        baseline_rho = bdata.get("baseline", {}).get("rho", "N/A")
        best = bdata.get("best")
        if best:
            print(f"  {bname:>12}: baseline={baseline_rho:.4f} → "
                  f"best={best['rho']:.4f} (delta={best['delta_rho']:+.4f}) "
                  f"@ layer {best['layer']}, alpha={best['alpha']:+.1f}")
        else:
            print(f"  {bname:>12}: baseline={baseline_rho} (no valid steering found)")

    print(f"\nResults: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract and evaluate behavioral steering vectors"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--behaviors", default="factual,sycophancy,bias",
        help="Comma-separated behaviors (default: factual,sycophancy,bias)",
    )
    parser.add_argument(
        "--method", default="mean_diff", choices=["mean_diff", "pca"],
        help="Steering vector extraction method",
    )
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer sweeps")
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    behaviors = [b.strip() for b in args.behaviors.split(",")]

    run_steering_experiment(
        model_id=args.model,
        behaviors=behaviors,
        device=args.device,
        method=args.method,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
