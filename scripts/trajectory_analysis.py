#!/usr/bin/env python3
"""Trajectory Analysis: Hidden-State Divergence + Logit Lens.

Tracks how internal representations evolve layer-by-layer for contrasting
text pairs (truthful vs sycophantic, factual vs false, etc.). Combines:

  1. Hidden-state trajectory divergence — cosine similarity between
     "good" and "bad" paths at each layer, revealing where reasoning drifts.

  2. Logit-lens projections — project each layer's hidden state to
     vocabulary space via the model's lm_head, showing what the model
     "thinks" at each depth. Reveals layers that "know" the answer but
     suppress it.

Works on MLX (Apple Silicon) and PyTorch (CUDA/CPU).

Usage:
    # Quick analysis on one behavior
    python scripts/trajectory_analysis.py Qwen/Qwen2.5-7B-Instruct \\
        --behaviors sycophancy --max-probes 20

    # Full analysis with logit lens
    python scripts/trajectory_analysis.py Qwen/Qwen2.5-7B-Instruct \\
        --behaviors sycophancy bias factual --max-probes 30

    # Specific layers only
    python scripts/trajectory_analysis.py Qwen/Qwen2.5-0.5B-Instruct \\
        --behaviors sycophancy --layers 0 6 12 18 23 --max-probes 10
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


# ── MLX All-Layer Hidden State Capture ──────────────────────────────────

_mlx_trajectory = {
    "states": {},     # layer_idx -> (B, seq_len, hidden_dim) mx.array
    "patched": False,
    "original_call": None,
    "active": False,  # Only capture when active
}


def _mlx_patch_model(model):
    """Patch the inner Qwen2Model to capture all-layer hidden states.

    Replaces model.model.__call__ to store h after each layer.
    """
    if _mlx_trajectory["patched"]:
        return

    inner_model = model.model
    inner_cls = inner_model.__class__
    original_call = inner_cls.__call__
    _mlx_trajectory["original_call"] = original_call

    # Import create_attention_mask from the model's module
    import importlib
    model_module = importlib.import_module(inner_cls.__module__)
    create_mask = model_module.create_attention_mask

    def patched_inner_call(self, inputs, cache=None, input_embeddings=None):
        import mlx.core as mx

        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_mask(h, cache[0])

        if _mlx_trajectory["active"]:
            _mlx_trajectory["states"][-1] = h  # Embedding layer

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask, c)
            if _mlx_trajectory["active"]:
                _mlx_trajectory["states"][i] = h

        return self.norm(h)

    inner_cls.__call__ = patched_inner_call
    _mlx_trajectory["patched"] = True


def _mlx_unpatch_model(model):
    """Restore the original inner model __call__."""
    if not _mlx_trajectory["patched"]:
        return
    inner_cls = model.model.__class__
    inner_cls.__call__ = _mlx_trajectory["original_call"]
    _mlx_trajectory["patched"] = False
    _mlx_trajectory["active"] = False
    _mlx_trajectory["states"] = {}


def _mlx_get_all_hidden_states(model, tokenizer, text, layers=None):
    """Run a forward pass and capture hidden states at all/specified layers.

    Args:
        model: MLX model.
        tokenizer: Tokenizer.
        text: Input text.
        layers: List of layer indices (None = all layers).

    Returns:
        dict[int, np.ndarray]: layer_idx -> (seq_len, hidden_dim) numpy array.
        Also returns token_ids as list[int].
    """
    import mlx.core as mx

    _mlx_patch_model(model)
    _mlx_trajectory["states"] = {}
    _mlx_trajectory["active"] = True

    tokens = tokenizer.encode(text)
    if len(tokens) > 512:
        tokens = tokens[:512]
    input_ids = mx.array([tokens])

    # Forward pass (captures all layer states)
    logits = model(input_ids)
    mx.eval(logits)

    _mlx_trajectory["active"] = False

    # Extract states at requested layers
    n_layers = len(model.model.layers)
    if layers is None:
        layers = list(range(n_layers))

    result = {}
    for li in layers:
        if li in _mlx_trajectory["states"]:
            h = _mlx_trajectory["states"][li]
            mx.eval(h)
            result[li] = np.array(h[0].astype(mx.float32))  # (seq_len, hidden_dim)

    # Also capture embedding layer if requested and available
    if -1 in _mlx_trajectory["states"] and -1 in (layers or [-1]):
        h = _mlx_trajectory["states"][-1]
        mx.eval(h)
        result[-1] = np.array(h[0].astype(mx.float32))

    # Clear to free memory
    _mlx_trajectory["states"] = {}

    return result, tokens


def _mlx_logit_lens(model, hidden_state):
    """Project a hidden state to vocabulary logits via lm_head.

    Args:
        model: MLX model.
        hidden_state: numpy array of shape (hidden_dim,) or (seq_len, hidden_dim).

    Returns:
        numpy array of logits (vocab_size,) or (seq_len, vocab_size).
    """
    import mlx.core as mx

    h = mx.array(hidden_state)
    if h.ndim == 1:
        h = h[None, :]  # (1, hidden_dim)
        squeeze = True
    else:
        squeeze = False

    # Apply final layer norm + lm_head
    h_normed = model.model.norm(h)
    if hasattr(model, "lm_head"):
        logits = model.lm_head(h_normed)
    elif model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(h_normed)
    else:
        raise RuntimeError("Cannot find lm_head")

    mx.eval(logits)
    result = np.array(logits.astype(mx.float32))

    if squeeze:
        result = result[0]
    return result


# ── PyTorch Variants ────────────────────────────────────────────────────

def _torch_get_all_hidden_states(model, tokenizer, text, layers=None, device="cpu"):
    """PyTorch version: get hidden states at specified layers."""
    import torch
    from rho_eval.interpretability.activation import LayerActivationCapture

    n_layers = model.config.num_hidden_layers
    if layers is None:
        layers = list(range(n_layers))

    cap = LayerActivationCapture(model, layers)

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=512).to(device)
        tokens = inputs["input_ids"][0].tolist()
        model(**inputs)

    result = {}
    for li in layers:
        h = cap.get(li)  # (1, seq_len, hidden_dim)
        result[li] = h[0].cpu().numpy()  # (seq_len, hidden_dim)

    cap.remove()
    return result, tokens


def _torch_logit_lens(model, hidden_state, device="cpu"):
    """PyTorch logit lens projection."""
    import torch

    with torch.no_grad():
        h = torch.tensor(hidden_state, dtype=torch.float32, device=device)
        if h.ndim == 1:
            h = h.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        h_normed = model.model.norm(h)
        logits = model.lm_head(h_normed)

        result = logits.cpu().numpy()
        if squeeze:
            result = result[0]
        return result


# ── Probe Loading ────────────────────────────────────────────────────────

def _load_contrast_pairs(behavior, max_probes=None):
    """Load contrast pairs for a behavior. Reuses head_mapping logic."""
    from rho_eval.interpretability.activation import build_contrast_pairs

    try:
        from rho_eval.behaviors import get_behavior
        behavior_plugin = get_behavior(behavior)
        probes = behavior_plugin.load_probes()
    except (KeyError, ValueError):
        from rho_eval.interpretability.subspaces import _load_probes_for_behavior
        probes = _load_probes_for_behavior(behavior)

    try:
        pairs = build_contrast_pairs(behavior, probes)
    except ValueError:
        return []

    if max_probes is not None and len(pairs) > max_probes:
        import random
        rng = random.Random(42)
        pairs = rng.sample(pairs, max_probes)

    return pairs


# ── Core Analysis ────────────────────────────────────────────────────────

def _cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _top_k_tokens(logits, tokenizer, k=5):
    """Get top-k token predictions from logits."""
    top_k_idx = np.argsort(logits)[-k:][::-1]
    results = []
    for idx in top_k_idx:
        token_str = tokenizer.decode([int(idx)])
        prob = float(np.exp(logits[idx] - np.max(logits)))  # Approximate softmax
        results.append({"token": token_str.strip(), "id": int(idx), "logit": float(logits[idx])})
    return results


def analyze_trajectory(
    model,
    tokenizer,
    backend,
    behaviors,
    layers=None,
    max_probes=None,
    device="cpu",
    verbose=True,
):
    """Analyze hidden-state trajectories and logit-lens projections.

    For each behavior's contrast pairs:
    1. Capture all-layer hidden states for positive (truthful) and
       negative (sycophantic/false/biased) texts.
    2. Compute per-layer cosine similarity between pos/neg last-token states.
    3. Apply logit-lens at each layer to see what the model "predicts" at depth.
    4. Identify divergence points where the paths separate.

    Returns:
        Dict with trajectory data, divergence analysis, and logit-lens results.
    """
    # Get model config
    if backend == "mlx":
        n_layers = len(model.model.layers)
        if hasattr(model, "args"):
            hidden_size = model.args.hidden_size
        else:
            hidden_size = 4096
    else:
        n_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size

    if layers is None:
        layers = list(range(n_layers))

    # Select extraction functions
    if backend == "mlx":
        get_states = lambda text: _mlx_get_all_hidden_states(
            model, tokenizer, text, layers
        )
        logit_lens = lambda h: _mlx_logit_lens(model, h)
    else:
        get_states = lambda text: _torch_get_all_hidden_states(
            model, tokenizer, text, layers, device
        )
        logit_lens = lambda h: _torch_logit_lens(model, h, device)

    if verbose:
        print(f"\n  Trajectory Analysis Configuration:")
        print(f"    Backend:    {backend}")
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

        # Collect per-layer stats across all pairs
        layer_cosines = {li: [] for li in layers}  # cosine sims
        layer_norm_ratios = {li: [] for li in layers}  # ||pos|| / ||neg||
        pos_logit_lens_agg = {li: [] for li in layers}  # top predictions
        neg_logit_lens_agg = {li: [] for li in layers}

        for pi, pair in enumerate(pairs):
            t0 = time.time()

            # Get all-layer hidden states for both texts
            pos_states, pos_tokens = get_states(pair["positive"])
            neg_states, neg_tokens = get_states(pair["negative"])

            for li in layers:
                if li not in pos_states or li not in neg_states:
                    continue

                # Last-token hidden state at this layer
                pos_h = pos_states[li][-1]  # (hidden_dim,)
                neg_h = neg_states[li][-1]  # (hidden_dim,)

                # Cosine similarity between pos and neg paths
                cos_sim = _cosine_sim(pos_h, neg_h)
                layer_cosines[li].append(cos_sim)

                # Norm ratio
                pos_norm = np.linalg.norm(pos_h)
                neg_norm = np.linalg.norm(neg_h)
                if neg_norm > 1e-10:
                    layer_norm_ratios[li].append(pos_norm / neg_norm)

                # Logit lens (sample first 3 pairs only — expensive)
                if pi < 3:
                    pos_logits = logit_lens(pos_h)
                    neg_logits = logit_lens(neg_h)
                    pos_logit_lens_agg[li].append(
                        _top_k_tokens(pos_logits, tokenizer, k=5)
                    )
                    neg_logit_lens_agg[li].append(
                        _top_k_tokens(neg_logits, tokenizer, k=5)
                    )

            if verbose and (pi + 1) % 5 == 0:
                elapsed = time.time() - t0
                print(f"    Pair {pi+1}/{len(pairs)} ({elapsed:.1f}s/pair)",
                      flush=True)

        # Compute statistics
        trajectory_data = {}
        for li in layers:
            cosines = layer_cosines[li]
            norms = layer_norm_ratios[li]
            if not cosines:
                continue

            trajectory_data[li] = {
                "cosine_mean": float(np.mean(cosines)),
                "cosine_std": float(np.std(cosines)),
                "cosine_min": float(np.min(cosines)),
                "norm_ratio_mean": float(np.mean(norms)) if norms else 1.0,
                "n_pairs": len(cosines),
            }

            # Add logit lens samples
            if pos_logit_lens_agg[li]:
                trajectory_data[li]["logit_lens_pos"] = pos_logit_lens_agg[li]
                trajectory_data[li]["logit_lens_neg"] = neg_logit_lens_agg[li]

        # Find divergence point (biggest cosine drop between consecutive layers)
        sorted_layers = sorted(trajectory_data.keys())
        max_drop = 0.0
        divergence_layer = None
        for i in range(1, len(sorted_layers)):
            prev = sorted_layers[i - 1]
            curr = sorted_layers[i]
            drop = trajectory_data[prev]["cosine_mean"] - trajectory_data[curr]["cosine_mean"]
            if drop > max_drop:
                max_drop = drop
                divergence_layer = curr

        # Early/late cosine comparison
        early_layers = sorted_layers[:len(sorted_layers)//3]
        late_layers = sorted_layers[-len(sorted_layers)//3:]
        early_cos = np.mean([trajectory_data[l]["cosine_mean"] for l in early_layers]) if early_layers else 0
        late_cos = np.mean([trajectory_data[l]["cosine_mean"] for l in late_layers]) if late_layers else 0

        results[behavior] = {
            "trajectory": {str(k): v for k, v in trajectory_data.items()},
            "divergence_layer": divergence_layer,
            "max_cosine_drop": float(max_drop),
            "early_cosine": float(early_cos),
            "late_cosine": float(late_cos),
            "drift_magnitude": float(early_cos - late_cos),
            "n_pairs": len(pairs),
        }

        if verbose:
            print(f"\n    Trajectory Summary ({behavior}):")
            print(f"      Early cosine (L0-{sorted_layers[len(sorted_layers)//3-1] if early_layers else '?'}): "
                  f"{early_cos:.4f}")
            print(f"      Late cosine  (L{sorted_layers[-len(sorted_layers)//3] if late_layers else '?'}-{sorted_layers[-1]}): "
                  f"{late_cos:.4f}")
            print(f"      Drift:       {early_cos - late_cos:+.4f}")
            if divergence_layer is not None:
                print(f"      Divergence:  Layer {divergence_layer} "
                      f"(drop={max_drop:.4f})")

    # Cleanup
    if backend == "mlx":
        _mlx_unpatch_model(model)

    return results


# ── Visualization ────────────────────────────────────────────────────────

def plot_trajectories(results, output_path):
    """Plot cosine similarity trajectories across layers."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available, skipping plot")
        return

    behaviors = list(results.keys())
    n = len(behaviors)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), squeeze=False)

    for bi, behavior in enumerate(behaviors):
        ax = axes[bi, 0]
        data = results[behavior]
        traj = data["trajectory"]

        layers_sorted = sorted([int(k) for k in traj.keys()])
        cosines = [traj[str(l)]["cosine_mean"] for l in layers_sorted]
        stds = [traj[str(l)]["cosine_std"] for l in layers_sorted]

        ax.plot(layers_sorted, cosines, "o-", color="steelblue",
                linewidth=2, markersize=4, label="mean cosine sim")
        ax.fill_between(
            layers_sorted,
            [c - s for c, s in zip(cosines, stds)],
            [c + s for c, s in zip(cosines, stds)],
            alpha=0.2, color="steelblue",
        )

        # Mark divergence point
        div_layer = data.get("divergence_layer")
        if div_layer is not None and str(div_layer) in traj:
            ax.axvline(x=div_layer, color="red", linestyle="--", alpha=0.7,
                       label=f"divergence (L{div_layer})")

        ax.set_title(f"{behavior} — Trajectory Divergence "
                     f"(drift={data['drift_magnitude']:+.4f})",
                     fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine Similarity (pos vs neg)")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="trajectory-analysis",
        description="Hidden-state trajectory divergence + logit-lens analysis",
    )

    parser.add_argument("model", help="HuggingFace model ID or local path")
    parser.add_argument(
        "--behaviors", nargs="+",
        default=["sycophancy", "factual", "bias"],
        help="Behaviors to analyze (default: sycophancy, factual, bias)",
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=None,
        help="Layer indices (default: all layers)",
    )
    parser.add_argument(
        "--max-probes", type=int, default=20,
        help="Max contrast pairs per behavior (default: 20)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "mlx", "cuda", "cpu"],
    )
    parser.add_argument("--no-plot", action="store_true")

    args = parser.parse_args(argv)

    if args.output is None:
        model_short = args.model.replace("/", "_").replace("\\", "_")
        args.output = f"results/trajectory/{model_short}"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"  Trajectory Analysis: Hidden States + Logit Lens")
    print(f"  Model: {args.model}")
    print(f"{'='*60}")

    # Load model
    print("\n  Loading model...", flush=True)
    from rho_eval.utils import load_model
    model, tokenizer, backend = load_model(args.model, device=args.device)
    print(f"  Loaded (backend={backend})", flush=True)

    # Run analysis
    results = analyze_trajectory(
        model, tokenizer, backend,
        behaviors=args.behaviors,
        layers=args.layers,
        max_probes=args.max_probes,
        device=args.device if args.device != "auto" else "cpu",
        verbose=True,
    )

    elapsed = time.time() - t_start

    # Save
    output = {
        "model": args.model,
        "backend": backend,
        "results": results,
        "elapsed_seconds": round(elapsed, 1),
    }
    json_path = output_dir / "trajectory_analysis.json"
    json_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Saved: {json_path}")

    if not args.no_plot and results:
        plot_path = output_dir / "trajectory_divergence.png"
        plot_trajectories(results, str(plot_path))

    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
