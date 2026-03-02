#!/usr/bin/env python3
"""Entropy Trajectory Analysis.

Tracks token-level entropy (uncertainty) and semantic entropy across
generated continuations, comparing "truthful" vs "sycophantic" paths.

Entropy spikes reveal confusion points where the model is uncertain
about which path to take. Early spikes = early distraction vulnerability.

Combines:
  1. Token-level entropy — H(p) at each generation step
  2. Logit-lens entropy — H(p) of the vocabulary projection at each LAYER
     for a given token position (how uncertain is each layer?)
  3. Per-layer entropy profiles — which layers are most/least certain?

Usage:
    python scripts/entropy_trajectory.py Qwen/Qwen2.5-7B-Instruct \\
        --behaviors sycophancy --max-probes 20

    python scripts/entropy_trajectory.py Qwen/Qwen2.5-0.5B-Instruct \\
        --behaviors sycophancy bias --max-probes 10
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Reuse trajectory infrastructure
from trajectory_analysis import (
    _mlx_patch_model,
    _mlx_unpatch_model,
    _mlx_get_all_hidden_states,
    _mlx_logit_lens,
    _load_contrast_pairs,
)


def _entropy(logits):
    """Compute entropy H(p) from logits (handles numerical stability)."""
    # Shift for numerical stability
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum()
    # Clip to avoid log(0)
    probs = np.clip(probs, 1e-10, 1.0)
    return float(-np.sum(probs * np.log2(probs)))


def analyze_entropy(
    model,
    tokenizer,
    backend,
    behaviors,
    layers=None,
    max_probes=None,
    verbose=True,
):
    """Analyze per-layer entropy profiles for contrasting text pairs.

    For each pair:
    1. Get hidden states at all layers for pos and neg texts.
    2. Project each layer's last-token state to vocab space via logit-lens.
    3. Compute entropy of the logit distribution at each layer.
    4. Compare entropy profiles between pos and neg.

    Returns:
        Dict with per-behavior entropy profiles and analysis.
    """
    if backend != "mlx":
        raise NotImplementedError("Entropy trajectory currently MLX-only")

    n_layers = len(model.model.layers)
    if layers is None:
        layers = list(range(n_layers))

    if hasattr(model, "args"):
        hidden_size = model.args.hidden_size
    else:
        hidden_size = 4096

    if verbose:
        print(f"\n  Entropy Analysis Configuration:")
        print(f"    Layers:     {len(layers)} of {n_layers}")
        print(f"    Hidden dim: {hidden_size}")
        print(f"    Behaviors:  {', '.join(behaviors)}")

    results = {}

    for behavior in behaviors:
        if verbose:
            print(f"\n  [{behavior}]", flush=True)

        pairs = _load_contrast_pairs(behavior, max_probes)
        if len(pairs) < 3:
            if verbose:
                print(f"    WARNING: Only {len(pairs)} pairs — skipping")
            continue

        if verbose:
            print(f"    {len(pairs)} contrast pairs", flush=True)

        # Per-layer entropy accumulators
        pos_entropies = {li: [] for li in layers}
        neg_entropies = {li: [] for li in layers}

        for pi, pair in enumerate(pairs):
            # Get all-layer hidden states
            pos_states, _ = _mlx_get_all_hidden_states(
                model, tokenizer, pair["positive"], layers
            )
            neg_states, _ = _mlx_get_all_hidden_states(
                model, tokenizer, pair["negative"], layers
            )

            for li in layers:
                if li not in pos_states or li not in neg_states:
                    continue

                # Logit-lens projection + entropy
                pos_logits = _mlx_logit_lens(model, pos_states[li][-1])
                neg_logits = _mlx_logit_lens(model, neg_states[li][-1])

                pos_entropies[li].append(_entropy(pos_logits))
                neg_entropies[li].append(_entropy(neg_logits))

            if verbose and (pi + 1) % 5 == 0:
                print(f"    Pair {pi+1}/{len(pairs)}", flush=True)

        # Compute statistics
        entropy_data = {}
        for li in layers:
            if not pos_entropies[li]:
                continue
            entropy_data[li] = {
                "pos_entropy_mean": float(np.mean(pos_entropies[li])),
                "pos_entropy_std": float(np.std(pos_entropies[li])),
                "neg_entropy_mean": float(np.mean(neg_entropies[li])),
                "neg_entropy_std": float(np.std(neg_entropies[li])),
                "delta_entropy": float(
                    np.mean(neg_entropies[li]) - np.mean(pos_entropies[li])
                ),
            }

        # Find peak entropy difference
        sorted_layers = sorted(entropy_data.keys())
        max_delta = 0.0
        peak_layer = None
        for li in sorted_layers:
            delta = abs(entropy_data[li]["delta_entropy"])
            if delta > max_delta:
                max_delta = delta
                peak_layer = li

        # Early vs late entropy
        early = sorted_layers[:len(sorted_layers)//3]
        late = sorted_layers[-len(sorted_layers)//3:]
        early_delta = np.mean([entropy_data[l]["delta_entropy"] for l in early]) if early else 0
        late_delta = np.mean([entropy_data[l]["delta_entropy"] for l in late]) if late else 0

        results[behavior] = {
            "entropy": {str(k): v for k, v in entropy_data.items()},
            "peak_delta_layer": peak_layer,
            "peak_delta": float(max_delta),
            "early_delta_mean": float(early_delta),
            "late_delta_mean": float(late_delta),
            "n_pairs": len(pairs),
        }

        if verbose:
            print(f"\n    Entropy Summary ({behavior}):")
            print(f"      Peak delta:  Layer {peak_layer} "
                  f"(delta={max_delta:+.3f} bits)")
            print(f"      Early delta: {early_delta:+.3f} bits")
            print(f"      Late delta:  {late_delta:+.3f} bits")
            # Show entropy curve at a few key layers
            for li in [sorted_layers[0], sorted_layers[len(sorted_layers)//2], sorted_layers[-1]]:
                d = entropy_data[li]
                print(f"      L{li:3d}: pos={d['pos_entropy_mean']:.2f}±{d['pos_entropy_std']:.2f}  "
                      f"neg={d['neg_entropy_mean']:.2f}±{d['neg_entropy_std']:.2f}  "
                      f"Δ={d['delta_entropy']:+.3f}")

    # Cleanup
    if backend == "mlx":
        _mlx_unpatch_model(model)

    return results


def plot_entropy(results, output_path):
    """Plot entropy trajectories."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available")
        return

    behaviors = list(results.keys())
    n = len(behaviors)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), squeeze=False)

    for bi, behavior in enumerate(behaviors):
        ax = axes[bi, 0]
        data = results[behavior]
        ent = data["entropy"]

        layers = sorted([int(k) for k in ent.keys()])
        pos_ent = [ent[str(l)]["pos_entropy_mean"] for l in layers]
        neg_ent = [ent[str(l)]["neg_entropy_mean"] for l in layers]
        pos_std = [ent[str(l)]["pos_entropy_std"] for l in layers]
        neg_std = [ent[str(l)]["neg_entropy_std"] for l in layers]

        ax.plot(layers, pos_ent, "o-", color="green", linewidth=2,
                markersize=3, label="positive (truthful)", alpha=0.8)
        ax.fill_between(layers,
                        [m-s for m, s in zip(pos_ent, pos_std)],
                        [m+s for m, s in zip(pos_ent, pos_std)],
                        alpha=0.15, color="green")

        ax.plot(layers, neg_ent, "s-", color="red", linewidth=2,
                markersize=3, label="negative (sycophantic)", alpha=0.8)
        ax.fill_between(layers,
                        [m-s for m, s in zip(neg_ent, neg_std)],
                        [m+s for m, s in zip(neg_ent, neg_std)],
                        alpha=0.15, color="red")

        # Mark peak delta
        peak = data.get("peak_delta_layer")
        if peak is not None:
            ax.axvline(x=peak, color="orange", linestyle="--", alpha=0.7,
                       label=f"peak delta (L{peak})")

        ax.set_title(f"{behavior} — Per-Layer Entropy "
                     f"(peak Δ={data['peak_delta']:+.3f} bits at L{peak})",
                     fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Entropy (bits)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {output_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="entropy-trajectory",
        description="Per-layer entropy analysis via logit-lens",
    )
    parser.add_argument("model", help="Model ID or path")
    parser.add_argument("--behaviors", nargs="+", default=["sycophancy"])
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--max-probes", type=int, default=20)
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "mlx", "cuda", "cpu"])
    parser.add_argument("--no-plot", action="store_true")

    args = parser.parse_args(argv)

    if args.output is None:
        model_short = args.model.replace("/", "_").replace("\\", "_")
        args.output = f"results/entropy/{model_short}"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"  Entropy Trajectory Analysis")
    print(f"  Model: {args.model}")
    print(f"{'='*60}")

    print("\n  Loading model...", flush=True)
    from rho_eval.utils import load_model
    model, tokenizer, backend = load_model(args.model, device=args.device)
    print(f"  Loaded (backend={backend})", flush=True)

    results = analyze_entropy(
        model, tokenizer, backend,
        behaviors=args.behaviors,
        layers=args.layers,
        max_probes=args.max_probes,
        verbose=True,
    )

    elapsed = time.time() - t_start

    output = {
        "model": args.model,
        "backend": backend,
        "results": results,
        "elapsed_seconds": round(elapsed, 1),
    }
    json_path = output_dir / "entropy_analysis.json"
    json_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Saved: {json_path}")

    if not args.no_plot and results:
        plot_path = output_dir / "entropy_trajectory.png"
        plot_entropy(results, str(plot_path))

    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
