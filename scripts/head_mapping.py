#!/usr/bin/env python3
"""Attention Head Behavioral Mapping.

Maps each attention head's contribution to each behavioral dimension by
measuring per-head activation differences between positive and negative
probe texts. Produces a full (n_layers × n_heads × n_behaviors) importance
tensor plus JSON output and optional heatmap visualization.

Works on any platform:
  - Apple Silicon: MLX (fast, native GPU)
  - CUDA: PyTorch + CUDA
  - CPU: PyTorch on CPU

Usage:
    # Full map (all layers, all behaviors with contrast pairs)
    python scripts/head_mapping.py Qwen/Qwen2.5-7B-Instruct

    # Quick scan (6 layers, 50 probes per behavior)
    python scripts/head_mapping.py Qwen/Qwen2.5-7B-Instruct --quick

    # Specific behaviors + layers
    python scripts/head_mapping.py Qwen/Qwen2.5-7B-Instruct \
        --behaviors sycophancy bias factual \
        --layers 6 12 17 24 28

    # Force device
    python scripts/head_mapping.py Qwen/Qwen2.5-7B-Instruct --device cpu
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


# ── MLX Head Output Extraction ──────────────────────────────────────────

# Module-level state for MLX attention capture (class-level patch requires this)
_mlx_capture = {"target": -1, "heads": None, "original_call": None, "patched": False}


def _mlx_patch_attention(model):
    """Patch the Attention class to capture per-head outputs.

    MLX modules don't support instance-level __call__ overrides or
    register_forward_hook. We patch the Attention class's __call__ to
    intercept the target layer's attention output before o_proj.

    The patch is idempotent and uses _mlx_capture['target'] to select
    which layer to capture from (-1 = none).
    """
    if _mlx_capture["patched"]:
        return

    attn_cls = model.model.layers[0].self_attn.__class__
    original_call = attn_cls.__call__
    _mlx_capture["original_call"] = original_call

    # Tag each attention instance with its layer index
    for i, layer in enumerate(model.model.layers):
        layer.self_attn._layer_idx = i

    # Import scaled_dot_product_attention from the model's module
    import importlib
    model_module = importlib.import_module(attn_cls.__module__)
    sdpa = model_module.scaled_dot_product_attention

    def patched_call(self, x, mask=None, cache=None):
        if getattr(self, "_layer_idx", -1) != _mlx_capture["target"]:
            return original_call(self, x, mask=mask, cache=cache)

        # Target layer: inline attention with capture
        import mlx.core as mx
        B, L, D = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = sdpa(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        # output: (B, n_heads, L, head_dim) -> (B, L, n_heads, head_dim)
        output_t = output.transpose(0, 2, 1, 3)
        _mlx_capture["heads"] = output_t  # Capture!
        output = output_t.reshape(B, L, -1)
        return self.o_proj(output)

    attn_cls.__call__ = patched_call
    _mlx_capture["patched"] = True


def _mlx_unpatch_attention(model):
    """Restore the original Attention.__call__."""
    if not _mlx_capture["patched"]:
        return
    attn_cls = model.model.layers[0].self_attn.__class__
    attn_cls.__call__ = _mlx_capture["original_call"]
    _mlx_capture["patched"] = False
    _mlx_capture["target"] = -1
    _mlx_capture["heads"] = None


def _mlx_extract_head_outputs(model, tokenizer, text, layer_idx):
    """Extract per-head outputs at a specific layer for an MLX model.

    Uses a class-level patch on the Attention module to capture per-head
    activations before the o_proj projection. Only the target layer
    is captured; all other layers run normally.

    Returns:
        numpy array of shape (n_heads, head_dim) — last-token head outputs.
    """
    import mlx.core as mx

    # Ensure patch is active
    _mlx_patch_attention(model)

    # Set target layer
    _mlx_capture["target"] = layer_idx
    _mlx_capture["heads"] = None

    # Tokenize and run forward pass
    tokens = tokenizer.encode(text)
    if len(tokens) > 512:
        tokens = tokens[:512]
    input_ids = mx.array([tokens])
    model(input_ids)

    if _mlx_capture["heads"] is None:
        raise RuntimeError(
            f"Failed to capture attention heads at layer {layer_idx}"
        )

    h = _mlx_capture["heads"]  # (B, L, n_heads, head_dim)
    mx.eval(h)

    # Extract last token, all heads -> (n_heads, head_dim)
    last_heads = np.array(h[0, -1, :, :].astype(mx.float32))

    # Clear to avoid holding memory
    _mlx_capture["heads"] = None

    return last_heads


def _torch_extract_head_outputs(model, tokenizer, text, layer_idx, device="cpu"):
    """Extract per-head outputs at a specific layer for a PyTorch model.

    Uses HeadOutputCapture hook from the interpretability module.

    Returns:
        numpy array of shape (n_heads, head_dim) — last-token head outputs.
    """
    import torch
    from rho_eval.interpretability.activation import HeadOutputCapture
    from rho_eval.interpretability.heads import _get_head_config

    n_heads, head_dim = _get_head_config(model)
    cap = HeadOutputCapture(model, layer_idx, n_heads, head_dim)

    try:
        with torch.no_grad():
            inputs = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=512,
            ).to(device)
            model(**inputs)
            h = cap.get()  # (1, n_heads, seq_len, head_dim)
            return h[0, :, -1, :].cpu().numpy()  # (n_heads, head_dim)
    finally:
        cap.remove()


# ── Probe Loading ────────────────────────────────────────────────────────

def _load_contrast_pairs(behavior, max_probes=None):
    """Load probes and build contrast pairs for a behavior.

    Uses the v2 behavior plugin system first, falling back to the legacy
    API. Only behaviors that produce valid contrast pairs are supported.

    Returns list of dicts with 'positive' and 'negative' keys.
    """
    from rho_eval.interpretability.activation import build_contrast_pairs

    # Try new plugin system first (handles refusal, deception, etc.)
    try:
        from rho_eval.behaviors import get_behavior
        behavior_plugin = get_behavior(behavior)
        probes = behavior_plugin.load_probes()
    except (KeyError, ValueError):
        # Fall back to legacy loader
        from rho_eval.interpretability.subspaces import _load_probes_for_behavior
        probes = _load_probes_for_behavior(behavior)

    try:
        pairs = build_contrast_pairs(behavior, probes)
    except ValueError:
        # This behavior doesn't support contrast pairs
        return []

    if max_probes is not None and len(pairs) > max_probes:
        import random
        rng = random.Random(42)
        pairs = rng.sample(pairs, max_probes)

    return pairs


# ── Core Mapping ─────────────────────────────────────────────────────────

def map_heads(
    model,
    tokenizer,
    backend,
    behaviors,
    layers=None,
    max_probes=None,
    device="cpu",
    verbose=True,
):
    """Map attention head importance across all behaviors and layers.

    For each (behavior, layer, head):
    1. Run positive probes, capture per-head last-token output.
    2. Run negative probes, capture per-head last-token output.
    3. importance = ||mean_pos_head - mean_neg_head|| / ||full_layer_diff||

    Args:
        model: Loaded model (MLX or PyTorch).
        tokenizer: Corresponding tokenizer.
        backend: "mlx" or "torch".
        behaviors: List of behavior names.
        layers: Layer indices (None = all layers).
        max_probes: Cap on probes per behavior.
        device: Torch device (ignored for MLX).
        verbose: Print progress.

    Returns:
        Dict with keys:
            - "head_map": dict[behavior][layer_idx] -> list of (head_idx, importance)
            - "metadata": model info, n_heads, n_layers, etc.
            - "raw_norms": dict[behavior][layer_idx] -> list of per-head norms
    """
    # Get model config
    if backend == "mlx":
        if hasattr(model, "args"):
            n_heads = model.args.num_attention_heads
            n_layers = len(model.model.layers)
            hidden_size = model.args.hidden_size
        elif hasattr(model, "config"):
            cfg = model.config
            if isinstance(cfg, dict):
                n_heads = cfg.get("num_attention_heads", 32)
                n_layers = len(model.model.layers)
                hidden_size = cfg.get("hidden_size", 4096)
            else:
                n_heads = getattr(cfg, "num_attention_heads", 32)
                n_layers = len(model.model.layers)
                hidden_size = getattr(cfg, "hidden_size", 4096)
        else:
            n_layers = len(model.model.layers)
            n_heads = 32
            hidden_size = 4096
    else:
        config = model.config
        n_heads = config.num_attention_heads
        n_layers = config.num_hidden_layers
        hidden_size = config.hidden_size

    head_dim = hidden_size // n_heads

    if layers is None:
        layers = list(range(n_layers))

    if verbose:
        print(f"\n  Head Mapping Configuration:")
        print(f"    Backend:    {backend}")
        print(f"    Layers:     {len(layers)} of {n_layers}")
        print(f"    Heads:      {n_heads} × {head_dim}d")
        print(f"    Behaviors:  {', '.join(behaviors)}")
        if max_probes:
            print(f"    Max probes: {max_probes}/behavior")

    # Extract function
    if backend == "mlx":
        extract_fn = lambda text, layer: _mlx_extract_head_outputs(
            model, tokenizer, text, layer
        )
    else:
        extract_fn = lambda text, layer: _torch_extract_head_outputs(
            model, tokenizer, text, layer, device=device
        )

    head_map = {}
    raw_norms = {}

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

        head_map[behavior] = {}
        raw_norms[behavior] = {}

        for li, layer_idx in enumerate(layers):
            t0 = time.time()

            pos_heads = []  # list of (n_heads, head_dim) arrays
            neg_heads = []

            for pair in pairs:
                pos_h = extract_fn(pair["positive"], layer_idx)
                neg_h = extract_fn(pair["negative"], layer_idx)
                pos_heads.append(pos_h)
                neg_heads.append(neg_h)

            # Stack: (n_pairs, n_heads, head_dim)
            pos_stack = np.stack(pos_heads)
            neg_stack = np.stack(neg_heads)

            # Per-head mean difference: (n_heads, head_dim)
            mean_pos = pos_stack.mean(axis=0)
            mean_neg = neg_stack.mean(axis=0)
            diff = mean_pos - mean_neg  # (n_heads, head_dim)

            # Per-head norms
            head_norms = np.linalg.norm(diff, axis=1)  # (n_heads,)

            # Full-layer diff norm (for normalization)
            full_norm = np.linalg.norm(diff.reshape(-1))

            if full_norm < 1e-10:
                importance = np.ones(n_heads) / n_heads
            else:
                importance = head_norms / full_norm

            # Store results
            head_map[behavior][layer_idx] = [
                {"head": int(h), "importance": float(importance[h]),
                 "norm": float(head_norms[h])}
                for h in range(n_heads)
            ]
            raw_norms[behavior][layer_idx] = head_norms.tolist()

            elapsed = time.time() - t0

            if verbose:
                # Show top-3 heads
                top3_idx = np.argsort(importance)[::-1][:3]
                top3_str = ", ".join(
                    f"h{i}={importance[i]:.3f}" for i in top3_idx
                )
                pct = (li + 1) / len(layers) * 100
                print(f"    L{layer_idx:3d}: top-3 [{top3_str}]  "
                      f"({elapsed:.1f}s, {pct:.0f}%)", flush=True)

    metadata = {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "layers_mapped": layers,
        "behaviors": behaviors,
        "max_probes": max_probes,
        "backend": backend,
    }

    # Clean up MLX patch if active
    if backend == "mlx":
        _mlx_unpatch_attention(model)

    return {
        "head_map": head_map,
        "raw_norms": raw_norms,
        "metadata": metadata,
    }


# ── Analysis ─────────────────────────────────────────────────────────────

def analyze_head_map(result):
    """Compute summary statistics from a head map.

    Returns dict with:
        - per_behavior_top_heads: top 10 heads per behavior (across all layers)
        - cross_behavior_density: heads that are important for multiple behaviors
        - layer_concentration: which layers have the most behavioral signal
        - specialist_heads: heads strongly aligned with one behavior
        - generalist_heads: heads important for many behaviors
    """
    head_map = result["head_map"]
    meta = result["metadata"]
    n_heads = meta["n_heads"]
    behaviors = list(head_map.keys())

    # Build importance tensor: (n_behaviors, n_layers_mapped, n_heads)
    layers = meta["layers_mapped"]
    imp_tensor = np.zeros((len(behaviors), len(layers), n_heads))

    for bi, behavior in enumerate(behaviors):
        for li, layer_idx in enumerate(layers):
            if layer_idx in head_map[behavior]:
                for entry in head_map[behavior][layer_idx]:
                    imp_tensor[bi, li, entry["head"]] = entry["importance"]

    analysis = {}

    # 1. Top heads per behavior (across all layers)
    per_behavior_top = {}
    for bi, behavior in enumerate(behaviors):
        flat_imp = []
        for li, layer_idx in enumerate(layers):
            for h in range(n_heads):
                flat_imp.append({
                    "layer": layer_idx,
                    "head": h,
                    "importance": float(imp_tensor[bi, li, h]),
                })
        flat_imp.sort(key=lambda x: x["importance"], reverse=True)
        per_behavior_top[behavior] = flat_imp[:10]
    analysis["per_behavior_top_heads"] = per_behavior_top

    # 2. Cross-behavior density: how many behaviors does each head serve?
    threshold = 1.0 / n_heads * 1.5  # 50% above uniform = "important"
    cross_density = {}
    for li, layer_idx in enumerate(layers):
        for h in range(n_heads):
            key = f"L{layer_idx}_H{h}"
            n_important = sum(
                1 for bi in range(len(behaviors))
                if imp_tensor[bi, li, h] > threshold
            )
            if n_important > 0:
                cross_density[key] = {
                    "layer": layer_idx,
                    "head": h,
                    "n_behaviors": n_important,
                    "behaviors": [
                        behaviors[bi] for bi in range(len(behaviors))
                        if imp_tensor[bi, li, h] > threshold
                    ],
                    "max_importance": float(imp_tensor[:, li, h].max()),
                }
    # Sort by n_behaviors desc, then max_importance desc
    cross_density_sorted = dict(sorted(
        cross_density.items(),
        key=lambda x: (x[1]["n_behaviors"], x[1]["max_importance"]),
        reverse=True,
    ))
    analysis["cross_behavior_density"] = cross_density_sorted

    # 3. Layer concentration: total behavioral signal per layer
    layer_signal = {}
    for li, layer_idx in enumerate(layers):
        total = float(imp_tensor[:, li, :].sum())
        layer_signal[layer_idx] = total
    analysis["layer_signal"] = layer_signal

    # 4. Specialist heads: high importance for exactly 1 behavior
    specialists = []
    for li, layer_idx in enumerate(layers):
        for h in range(n_heads):
            importances = imp_tensor[:, li, h]
            above_threshold = importances > threshold
            if above_threshold.sum() == 1:
                bi = np.argmax(importances)
                if importances[bi] > threshold * 2:  # strongly specialist
                    specialists.append({
                        "layer": layer_idx,
                        "head": h,
                        "behavior": behaviors[bi],
                        "importance": float(importances[bi]),
                    })
    specialists.sort(key=lambda x: x["importance"], reverse=True)
    analysis["specialist_heads"] = specialists[:20]

    # 5. Generalist heads: above threshold for 3+ behaviors
    generalists = [
        v for v in cross_density_sorted.values()
        if v["n_behaviors"] >= 3
    ]
    analysis["generalist_heads"] = generalists[:20]

    return analysis


# ── Visualization ────────────────────────────────────────────────────────

def plot_head_map(result, output_path):
    """Generate a heatmap visualization of head importance.

    Creates a grid of (layers × heads) for each behavior, plus an
    aggregate cross-behavior density plot.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError:
        print("  WARNING: matplotlib not available, skipping visualization")
        return

    head_map = result["head_map"]
    meta = result["metadata"]
    behaviors = list(head_map.keys())
    layers = meta["layers_mapped"]
    n_heads = meta["n_heads"]

    n_behaviors = len(behaviors)
    fig, axes = plt.subplots(
        n_behaviors + 1, 1,
        figsize=(max(12, n_heads * 0.5), (n_behaviors + 1) * max(3, len(layers) * 0.3)),
        squeeze=False,
    )

    # Per-behavior heatmaps
    for bi, behavior in enumerate(behaviors):
        ax = axes[bi, 0]
        matrix = np.zeros((len(layers), n_heads))
        for li, layer_idx in enumerate(layers):
            if layer_idx in head_map[behavior]:
                for entry in head_map[behavior][layer_idx]:
                    matrix[li, entry["head"]] = entry["importance"]

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                       norm=Normalize(vmin=0, vmax=matrix.max() * 1.1))
        ax.set_title(f"{behavior}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Layer")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, fontsize=7)
        if bi == n_behaviors - 1:
            ax.set_xlabel("Head")
        ax.set_xticks(range(0, n_heads, max(1, n_heads // 10)))
        plt.colorbar(im, ax=ax, shrink=0.8, label="importance")

    # Aggregate: cross-behavior density
    ax = axes[n_behaviors, 0]
    agg_matrix = np.zeros((len(layers), n_heads))
    threshold = 1.0 / n_heads * 1.5
    for li, layer_idx in enumerate(layers):
        for h in range(n_heads):
            count = 0
            for behavior in behaviors:
                if layer_idx in head_map[behavior]:
                    for entry in head_map[behavior][layer_idx]:
                        if entry["head"] == h and entry["importance"] > threshold:
                            count += 1
            agg_matrix[li, h] = count

    im = ax.imshow(agg_matrix, aspect="auto", cmap="viridis",
                   norm=Normalize(vmin=0, vmax=max(1, agg_matrix.max())))
    ax.set_title("Cross-Behavior Density (# behaviors above threshold)",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Head")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=7)
    ax.set_xticks(range(0, n_heads, max(1, n_heads // 10)))
    plt.colorbar(im, ax=ax, shrink=0.8, label="# behaviors")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="head-mapping",
        description="Map attention head behavioral roles across all layers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full map (all layers, all supported behaviors)
  python scripts/head_mapping.py Qwen/Qwen2.5-7B-Instruct

  # Quick scan (6 auto-selected layers, 50 probes/behavior)
  python scripts/head_mapping.py Qwen/Qwen2.5-7B-Instruct --quick

  # Specific behaviors and layers
  python scripts/head_mapping.py Qwen/Qwen2.5-7B-Instruct \\
      --behaviors sycophancy bias --layers 6 12 17 24

  # Save to specific directory
  python scripts/head_mapping.py Qwen/Qwen2.5-7B-Instruct -o results/head_map/
""",
    )

    parser.add_argument("model", help="HuggingFace model ID or local path")
    parser.add_argument(
        "--behaviors", nargs="+",
        default=["factual", "sycophancy", "bias", "toxicity"],
        help="Behaviors to map (default: factual, sycophancy, bias, toxicity). "
             "Also supports refusal, deception if probes available.",
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=None,
        help="Layer indices to map (default: all layers)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 6 auto-selected layers, 50 probes/behavior",
    )
    parser.add_argument(
        "--max-probes", type=int, default=None,
        help="Max contrast pairs per behavior (default: all)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output directory (default: results/head_map/<model>/)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "mlx", "cuda", "cpu"],
        help="Backend (default: auto)",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip heatmap visualization",
    )

    args = parser.parse_args(argv)

    # Quick mode overrides
    if args.quick:
        if args.max_probes is None:
            args.max_probes = 50
        # layers set after model load (need n_layers)

    # Output directory
    if args.output is None:
        model_short = args.model.replace("/", "_").replace("\\", "_")
        args.output = f"results/head_map/{model_short}"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"  Attention Head Behavioral Mapping")
    print(f"  Model:      {args.model}")
    print(f"  Behaviors:  {', '.join(args.behaviors)}")
    print(f"  Device:     {args.device}")
    print(f"  Quick mode: {args.quick}")
    print(f"{'='*60}")

    # ── Load model ──────────────────────────────────────────────
    print("\n  Loading model...", flush=True)
    from rho_eval.utils import load_model
    model, tokenizer, backend = load_model(args.model, device=args.device)
    print(f"  Loaded (backend={backend})", flush=True)

    # Get n_layers for quick mode
    if backend == "mlx":
        n_layers = len(model.model.layers)
    else:
        n_layers = model.config.num_hidden_layers

    if args.quick and args.layers is None:
        # Auto-select ~6 layers at standard depth percentages
        pcts = [0.25, 0.375, 0.50, 0.625, 0.75, 0.875]
        args.layers = sorted(set(
            max(0, min(n_layers - 1, int(pct * n_layers)))
            for pct in pcts
        ))
        print(f"  Quick mode: selected layers {args.layers}")

    # ── Map heads ──────────────────────────────────────────────
    result = map_heads(
        model, tokenizer, backend,
        behaviors=args.behaviors,
        layers=args.layers,
        max_probes=args.max_probes,
        device=args.device if args.device != "auto" else "cpu",
        verbose=True,
    )

    # Add model name to metadata
    result["metadata"]["model"] = args.model

    # ── Analysis ──────────────────────────────────────────────
    print("\n  Analyzing head map...", flush=True)
    analysis = analyze_head_map(result)

    # Print summary
    print(f"\n  {'='*55}")
    print(f"  SUMMARY")
    print(f"  {'='*55}")

    # Top specialist heads
    print(f"\n  Top Specialist Heads (strong for 1 behavior):")
    for s in analysis["specialist_heads"][:10]:
        print(f"    L{s['layer']:3d} H{s['head']:2d}: "
              f"{s['behavior']:<12s} importance={s['importance']:.3f}")

    # Generalist heads
    if analysis["generalist_heads"]:
        print(f"\n  Generalist Heads (important for 3+ behaviors):")
        for g in analysis["generalist_heads"][:10]:
            print(f"    L{g['layer']:3d} H{g['head']:2d}: "
                  f"{g['n_behaviors']} behaviors "
                  f"({', '.join(g['behaviors'])})")

    # Layer signal
    print(f"\n  Layer Signal Concentration:")
    signal = analysis["layer_signal"]
    top_layers = sorted(signal.items(), key=lambda x: x[1], reverse=True)[:5]
    for layer_idx, total in top_layers:
        print(f"    Layer {layer_idx:3d}: total_signal={total:.3f}")

    # ── Save ──────────────────────────────────────────────────
    elapsed = time.time() - t_start

    # Save full result
    # Convert int keys to strings for JSON serialization
    serializable_map = {}
    for behavior, layer_data in result["head_map"].items():
        serializable_map[behavior] = {
            str(k): v for k, v in layer_data.items()
        }
    serializable_norms = {}
    for behavior, layer_data in result["raw_norms"].items():
        serializable_norms[behavior] = {
            str(k): v for k, v in layer_data.items()
        }

    output = {
        "metadata": result["metadata"],
        "head_map": serializable_map,
        "raw_norms": serializable_norms,
        "analysis": {
            "per_behavior_top_heads": analysis["per_behavior_top_heads"],
            "specialist_heads": analysis["specialist_heads"],
            "generalist_heads": analysis["generalist_heads"],
            "layer_signal": {str(k): v for k, v in analysis["layer_signal"].items()},
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    json_path = output_dir / "head_map.json"
    json_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Saved: {json_path}")

    # Save analysis separately for quick reference
    analysis_path = output_dir / "head_analysis.json"
    analysis_serializable = {
        "per_behavior_top_heads": analysis["per_behavior_top_heads"],
        "specialist_heads": analysis["specialist_heads"],
        "generalist_heads": analysis["generalist_heads"],
        "layer_signal": {str(k): v for k, v in analysis["layer_signal"].items()},
        "cross_behavior_density": {
            k: v for k, v in list(analysis["cross_behavior_density"].items())[:50]
        },
    }
    analysis_path.write_text(json.dumps(analysis_serializable, indent=2))
    print(f"  Saved: {analysis_path}")

    # Visualization
    if not args.no_plot:
        plot_path = output_dir / "head_map.png"
        plot_head_map(result, str(plot_path))

    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
