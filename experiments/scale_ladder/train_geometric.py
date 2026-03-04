#!/usr/bin/env python3
"""Train a small causal LM with Grassmann geometric regularization.

Phase 3 of "Designed Geometry": tests whether providing the model with a
truer structural prior during training improves behavioral outcomes.

**Core idea: accounting for the surprising truth.**
Models fail exactly where truth is surprising. Next-token prediction converges
on expected continuations, not true ones — it minimizes loss over training data
including its errors (frequency bias, human false beliefs, agreeable-over-honest
patterns). Surprising truths have low frequency by definition, so they lose.

The geometric regularizer gives the model structural room to hold surprising
truths independently. When behavioral subspaces are entangled (~50° angles),
a strong prior in one dimension overwrites truth in another. When they're
orthogonal (~90°), each dimension can represent its own surprising truths
without interference. This is Bayes' theorem at the weight level: a truer
structural prior (separated subspaces) produces a more accurate posterior
(behavioral emergence that otherwise wouldn't happen at this scale).

**Hypothesis:** Behavioral emergence (bias ρ, sycophancy ρ) at 7M is blocked
by entangled subspace geometry (~50° mean angle). Geometric regularization
during training can widen angles to ~60°+, creating the structural scaffold
that allows behavioral signal to crystallize.

**Prediction:** bias ρ > 0 at 7M with geometric regularization (vs ρ = 0.0
in all vanilla 7M checkpoints).

**Required controls** (to distinguish geometry from confounds):
1. Content-exposure control: probe texts as extra LM data, no geometric loss.
   Tests whether text exposure alone causes emergence.
2. Scrambled-geometry control: same probe forward passes, penalize within-behavior
   similarity instead of cross-behavior. Same compute, no geometric signal.
3. Held-out evaluation: regularize on 50% of probes, measure ρ on the other 50%.
   Tests whether the effect is structural or memorization.

**Method:**
- Standard LM training (cross-entropy) + periodic geometric loss
- Geometric loss: penalizes cosine similarity between behavioral
  mean-difference vectors (steering vector proxies) below an angle floor
- Computed every `geo_interval` steps on small samples of contrast pairs
- Full subspace audits saved periodically for comparison with vanilla trajectory

Usage:
    # Standard run (7M with geometric regularization)
    python experiments/scale_ladder/train_geometric.py --size 7M --seed 42 --device mps

    # Custom parameters
    python experiments/scale_ladder/train_geometric.py --size 7M --seed 42 --device cpu \\
        --geo-weight 1.0 --angle-floor 60 --geo-interval 50

    # Control run (no geometric loss, same code path)
    python experiments/scale_ladder/train_geometric.py --size 7M --seed 42 --geo-weight 0

    # With explicit checkpoints matching vanilla hires for direct comparison
    python experiments/scale_ladder/train_geometric.py --size 7M --seed 42 --device mps \\
        --checkpoint-steps 800 1000 1200 1400 1600 1800 2000 2200 2400 2800 3200 4000 4880
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale_ladder.configs import SCALE_CONFIGS, SCALE_ORDER, SUBSPACE_BEHAVIORS
from experiments.scale_ladder.train_model import (
    build_model,
    build_tokenizer,
    packed_token_iterator,
)


# ── Behavioral contrast pair loading ──────────────────────────────


def load_all_contrast_pairs(behaviors: list[str]) -> dict[str, list[dict]]:
    """Pre-load contrast pairs for all behaviors (done once at startup)."""
    from rho_eval.interpretability.activation import build_contrast_pairs
    from rho_eval.interpretability.subspaces import _load_probes_for_behavior

    all_pairs = {}
    for b in behaviors:
        probes = _load_probes_for_behavior(b)
        pairs = build_contrast_pairs(b, probes)
        all_pairs[b] = pairs
        print(f"    {b}: {len(pairs)} contrast pairs")
    return all_pairs


# ── Differentiable geometric loss ─────────────────────────────────


def _forward_capture(model, model_layers, tokenizer, text, layers, device):
    """Forward pass capturing last-token hidden states with gradient tracking.

    Returns dict mapping layer_idx → tensor(d_model) connected to the
    computation graph for backprop.
    """
    hooks = []
    activations = {}

    def make_hook(layer_idx):
        def hook_fn(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            activations[layer_idx] = h[:, -1, :]  # (1, d_model)

        return hook_fn

    for l in layers:
        hooks.append(model_layers[l].register_forward_hook(make_hook(l)))

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256
    ).to(device)
    model(input_ids=inputs["input_ids"])

    for h in hooks:
        h.remove()

    return {l: activations[l].squeeze(0) for l in layers}


def compute_geometric_loss(
    model,
    model_layers,
    tokenizer,
    contrast_pairs: dict[str, list[dict]],
    behaviors: list[str],
    layers: list[int],
    device: torch.device,
    n_pairs: int = 20,
    angle_floor: float = 60.0,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, dict]:
    """Differentiable loss penalizing behavioral subspace entanglement.

    For each behavior:
      1. Sample n_pairs contrast pairs
      2. Forward pass both positive and negative texts
      3. Compute mean activation difference at each layer (steering vector proxy)

    For each (behavior_i, behavior_j) pair at each layer:
      4. Compute cosine similarity between steering vectors
      5. Penalize: relu(|cos_sim| - cos(angle_floor))^2

    Returns (loss_tensor, diagnostics_dict).
    Gradients flow through the model for parameter updates.
    """
    cos_floor = math.cos(math.radians(angle_floor))
    if rng is None:
        rng = random.Random(42)

    # Compute steering vectors per behavior per layer
    steering_vecs = {}  # behavior → {layer_idx → tensor(d_model)}

    for behavior in behaviors:
        pairs = contrast_pairs.get(behavior, [])
        if len(pairs) < 2:
            continue
        sampled = rng.sample(pairs, min(n_pairs, len(pairs)))

        layer_diffs = {l: [] for l in layers}
        for pair in sampled:
            pos_acts = _forward_capture(
                model, model_layers, tokenizer, pair["positive"], layers, device
            )
            neg_acts = _forward_capture(
                model, model_layers, tokenizer, pair["negative"], layers, device
            )
            for l in layers:
                layer_diffs[l].append(pos_acts[l] - neg_acts[l])

        steering_vecs[behavior] = {
            l: torch.stack(layer_diffs[l]).mean(dim=0) for l in layers
        }

    # Pairwise orthogonality penalty
    penalties = []
    angle_log = {}

    available = [b for b in behaviors if b in steering_vecs]
    for l in layers:
        for i, b1 in enumerate(available):
            for j in range(i + 1, len(available)):
                b2 = available[j]
                v1 = steering_vecs[b1][l]
                v2 = steering_vecs[b2][l]

                cos_sim = F.cosine_similarity(
                    v1.unsqueeze(0), v2.unsqueeze(0)
                ).squeeze()
                penalty = F.relu(cos_sim.abs() - cos_floor) ** 2
                penalties.append(penalty)

                # Log proxy angle (detached, for monitoring only)
                with torch.no_grad():
                    abs_cos = min(1.0, max(0.0, abs(cos_sim.item())))
                    angle_deg = math.degrees(math.acos(abs_cos))
                    key = f"L{l}_{b1[:4]}_{b2[:4]}"
                    angle_log[key] = round(angle_deg, 1)

    if penalties:
        loss = torch.stack(penalties).mean()
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)

    diag = {
        "n_terms": len(penalties),
        "angles": angle_log,
        "mean_angle": round(
            sum(angle_log.values()) / max(len(angle_log), 1), 1
        ),
        "loss": round(loss.item(), 6),
    }
    return loss, diag


# ── Full subspace audit (monitoring, no grad) ─────────────────────


def run_subspace_audit(
    model, tokenizer, layers, device, behaviors, max_probes=50
) -> dict:
    """Full SVD-based subspace extraction + overlap for ground-truth monitoring.

    Uses the same methodology as scale_audit.py but lighter (fewer probes).
    Switches to eval mode and uses no_grad for efficiency.
    """
    from rho_eval.interpretability.overlap import compute_overlap
    from rho_eval.interpretability.subspaces import extract_subspaces

    was_training = model.training
    model.eval()
    with torch.no_grad():
        subspaces = extract_subspaces(
            model,
            tokenizer,
            behaviors,
            layers=layers,
            device=str(device),
            max_rank=50,
            max_probes=max_probes,
            verbose=False,
        )
        overlap = compute_overlap(subspaces, top_k=10, verbose=False)
    if was_training:
        model.train()

    # Extract key metrics
    result = {"effective_dims": {}, "angles": {}}

    for behavior, layer_results in subspaces.items():
        result["effective_dims"][behavior] = {
            str(l): sr.effective_dim for l, sr in layer_results.items()
        }

    for layer_idx, om in overlap.items():
        n = len(om.behaviors)
        off_diag = []
        pair_angles = {}
        for i in range(n):
            for j in range(i + 1, n):
                angle = om.subspace_angles[i][j]
                off_diag.append(angle)
                pair_angles[
                    f"{om.behaviors[i][:4]}_{om.behaviors[j][:4]}"
                ] = angle

        result["angles"][str(layer_idx)] = {
            "mean": round(
                sum(off_diag) / len(off_diag), 2
            )
            if off_diag
            else 0.0,
            "pairs": pair_angles,
        }

    all_means = [v["mean"] for v in result["angles"].values()]
    result["overall_mean_angle"] = round(
        sum(all_means) / max(len(all_means), 1), 2
    )

    return result


# ── Main training loop ────────────────────────────────────────────


def train_geometric(
    size: str,
    seed: int = 42,
    device_str: str = "cpu",
    output_dir: str | None = None,
    dataset: str = "openwebtext",
    geo_weight: float = 1.0,
    geo_interval: int = 50,
    geo_pairs: int = 20,
    angle_floor: float = 60.0,
    audit_interval: int = 500,
    checkpoint_steps: list[int] | None = None,
    behaviors: list[str] | None = None,
):
    """Train a model with geometric regularization at the given scale."""
    if size not in SCALE_CONFIGS:
        print(f"  ERROR: Unknown size '{size}'. Choose from: {SCALE_ORDER}")
        return 1

    cfg = SCALE_CONFIGS[size]
    geo_enabled = geo_weight > 0

    # Output directory
    if output_dir is None:
        tag = f"geo_w{geo_weight}_f{int(angle_floor)}" if geo_enabled else "geo_control"
        output_dir = f"results/scale_ladder/{size}_seed{seed}_{tag}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"  Geometric Training — {size}")
    print(f"  Config: {cfg['n_layer']}L, {cfg['n_head']}H, d={cfg['n_embd']}")
    print(f"  Target: {cfg['train_tokens']:,} tokens")
    if geo_enabled:
        print(f"  Geometric: weight={geo_weight}, floor={angle_floor}°, "
              f"interval={geo_interval}, pairs={geo_pairs}")
    else:
        print(f"  Geometric: DISABLED (control run)")
    print(f"  Device: {device_str}, Seed: {seed}")
    print(f"{'='*60}\n")

    torch.manual_seed(seed)
    rng = random.Random(seed)

    # Build model
    print("  Building model...", flush=True)
    model, n_params = build_model(cfg)
    device = torch.device(device_str)
    model = model.to(device)
    model.train()

    # Build tokenizer
    tokenizer = build_tokenizer()

    # Get model layers for hook registration
    from rho_eval.utils import get_layers

    model_layers = get_layers(model)
    all_layer_indices = list(range(cfg["n_layer"]))

    # Pre-load behavioral contrast pairs
    if behaviors is None:
        behaviors = list(SUBSPACE_BEHAVIORS)

    if geo_enabled:
        print("  Loading contrast pairs...", flush=True)
        contrast_pairs = load_all_contrast_pairs(behaviors)
    else:
        contrast_pairs = {}

    # Optimizer (same as train_model.py)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # LR schedule: linear warmup then cosine decay
    block_size = cfg["n_positions"]
    batch_size = cfg["batch_size"]
    tokens_per_step = block_size * batch_size
    total_steps = cfg["train_tokens"] // tokens_per_step
    warmup_steps = min(500, total_steps // 10)
    min_lr = cfg["lr"] * 0.1

    def get_lr(step):
        if step < warmup_steps:
            return cfg["lr"] * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr + 0.5 * (cfg["lr"] - min_lr) * (
            1 + math.cos(math.pi * progress)
        )

    print(f"  Total steps: {total_steps:,} ({warmup_steps} warmup)")
    print(f"  Tokens/step: {tokens_per_step:,}\n", flush=True)

    # Data stream
    data_iter = packed_token_iterator(
        tokenizer, block_size, seed=seed, dataset=dataset
    )

    # Training history
    loss_history = []
    geo_history = []
    audit_history = []
    checkpoint_set = set(checkpoint_steps) if checkpoint_steps else set()
    log_interval = 100
    save_interval = max(total_steps // 5, 1000)

    # ── Initial subspace audit ──
    print("  Running initial subspace audit (step 0)...", flush=True)
    audit_0 = run_subspace_audit(
        model, tokenizer, all_layer_indices, device, behaviors
    )
    audit_0["step"] = 0
    audit_history.append(audit_0)
    (output_path / "audit_step_0").mkdir(exist_ok=True)
    (output_path / "audit_step_0" / "audit_data.json").write_text(
        json.dumps(audit_0, indent=2)
    )
    print(f"    Mean angle: {audit_0['overall_mean_angle']}°\n", flush=True)

    # ── Training loop ──
    best_loss = float("inf")
    last_geo_diag = None

    for step in range(total_steps):
        # Collect LM batch
        batch_tokens = []
        for _ in range(batch_size):
            try:
                batch_tokens.append(next(data_iter))
            except StopIteration:
                data_iter = packed_token_iterator(
                    tokenizer, block_size, seed=seed + step, dataset=dataset
                )
                batch_tokens.append(next(data_iter))

        input_ids = torch.tensor(batch_tokens, dtype=torch.long, device=device)

        # Update LR
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward LM loss
        outputs = model(input_ids=input_ids, labels=input_ids.clone())
        lm_loss = outputs.loss

        # Geometric loss (every geo_interval steps)
        geo_loss_val = 0.0
        if geo_enabled and (step + 1) % geo_interval == 0:
            geo_loss, geo_diag = compute_geometric_loss(
                model,
                model_layers,
                tokenizer,
                contrast_pairs,
                behaviors,
                all_layer_indices,
                device,
                n_pairs=geo_pairs,
                angle_floor=angle_floor,
                rng=rng,
            )
            total_loss = lm_loss + geo_weight * geo_loss
            geo_loss_val = geo_loss.item()
            last_geo_diag = geo_diag
            geo_history.append({"step": step + 1, **geo_diag})
        else:
            total_loss = lm_loss

        # Backward + optimize
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Track
        tokens_seen = (step + 1) * tokens_per_step
        lm_val = lm_loss.item()
        if lm_val < best_loss:
            best_loss = lm_val

        loss_history.append(
            {
                "step": step + 1,
                "lm_loss": lm_val,
                "geo_loss": geo_loss_val,
                "lr": lr,
                "tokens": tokens_seen,
            }
        )

        # ── Logging ──
        if (step + 1) % log_interval == 0:
            elapsed = time.time() - t_start
            tok_per_sec = tokens_seen / elapsed
            geo_str = ""
            if last_geo_diag:
                geo_str = (
                    f" | geo={last_geo_diag['loss']:.4f} "
                    f"(∠={last_geo_diag['mean_angle']}°)"
                )
            print(
                f"  step {step+1:>6d}/{total_steps} | "
                f"lm={lm_val:.4f}{geo_str} | lr={lr:.2e} | "
                f"{tok_per_sec:,.0f} tok/s | {elapsed/60:.1f}min",
                flush=True,
            )

        # ── Full subspace audit ──
        if (step + 1) % audit_interval == 0:
            print(
                f"  [audit] Subspace audit at step {step+1}...",
                flush=True,
            )
            audit = run_subspace_audit(
                model, tokenizer, all_layer_indices, device, behaviors
            )
            audit["step"] = step + 1
            audit_history.append(audit)

            audit_dir = output_path / f"audit_step_{step+1}"
            audit_dir.mkdir(exist_ok=True)
            (audit_dir / "audit_data.json").write_text(
                json.dumps(audit, indent=2)
            )
            print(
                f"    Mean angle: {audit['overall_mean_angle']}°",
                flush=True,
            )

        # ── Checkpoints ──
        save_regular = (step + 1) % save_interval == 0
        save_explicit = (step + 1) in checkpoint_set
        if save_regular or save_explicit:
            ckpt_path = output_path / f"checkpoint_{step+1}"
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            if save_explicit:
                print(
                    f"  [ckpt] Saved step {step+1} → {ckpt_path}",
                    flush=True,
                )

    # ── Final model + audit ──
    print(f"\n  Saving final model...", flush=True)
    model_path = output_path / "model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    print("  Running final subspace audit...", flush=True)
    final_audit = run_subspace_audit(
        model, tokenizer, all_layer_indices, device, behaviors
    )
    final_audit["step"] = total_steps
    audit_history.append(final_audit)

    audit_dir = output_path / f"audit_step_{total_steps}"
    audit_dir.mkdir(exist_ok=True)
    (audit_dir / "audit_data.json").write_text(
        json.dumps(final_audit, indent=2)
    )
    print(
        f"    Final mean angle: {final_audit['overall_mean_angle']}°\n",
        flush=True,
    )

    # ── Save training metrics ──
    elapsed = time.time() - t_start
    tokens_seen = total_steps * tokens_per_step

    metrics = {
        "size": size,
        "seed": seed,
        "device": device_str,
        "n_params": n_params,
        "config": cfg,
        "geometric": {
            "enabled": geo_enabled,
            "weight": geo_weight,
            "interval": geo_interval,
            "pairs": geo_pairs,
            "angle_floor": angle_floor,
            "behaviors": behaviors,
        },
        "total_steps": total_steps,
        "tokens_seen": tokens_seen,
        "final_lm_loss": loss_history[-1]["lm_loss"] if loss_history else None,
        "best_lm_loss": best_loss,
        "elapsed_seconds": round(elapsed, 1),
        "tokens_per_second": round(tokens_seen / elapsed, 1),
        "initial_mean_angle": audit_history[0]["overall_mean_angle"],
        "final_mean_angle": audit_history[-1]["overall_mean_angle"],
        # Subsample histories for reasonable file size
        "loss_history": loss_history[:: max(len(loss_history) // 200, 1)],
        "geo_history": geo_history,
        "audit_history": audit_history,
    }

    metrics_path = output_path / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"  Saved metrics: {metrics_path}")

    print(f"\n  Training complete!")
    print(f"  Final LM loss: {metrics['final_lm_loss']:.4f}")
    print(
        f"  Angle trajectory: {metrics['initial_mean_angle']}° → "
        f"{metrics['final_mean_angle']}°"
    )
    print(f"  Time: {elapsed/60:.1f} min ({elapsed/3600:.1f}h)")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    return 0


# ── CLI ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        prog="train-geometric",
        description=(
            "Train a causal LM with Grassmann geometric regularization. "
            "Tests whether enforcing subspace orthogonality enables "
            "behavioral emergence at smaller scales."
        ),
    )
    parser.add_argument(
        "--size",
        required=True,
        choices=SCALE_ORDER,
        help="Model size (e.g., 7M, 18M, 64M)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "cuda"],
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output directory (default: results/scale_ladder/{size}_seed{seed}_geo_...)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="openwebtext",
        choices=["openwebtext", "fineweb-edu", "wikitext"],
    )

    # Geometric regularization parameters
    geo = parser.add_argument_group("geometric regularization")
    geo.add_argument(
        "--geo-weight",
        type=float,
        default=1.0,
        help="Weight for geometric orthogonality loss (0 = disabled/control)",
    )
    geo.add_argument(
        "--geo-interval",
        type=int,
        default=50,
        help="Compute geometric loss every N steps (default: 50)",
    )
    geo.add_argument(
        "--geo-pairs",
        type=int,
        default=20,
        help="Contrast pairs sampled per behavior for geometric loss (default: 20)",
    )
    geo.add_argument(
        "--angle-floor",
        type=float,
        default=60.0,
        help="Target minimum angle (degrees) between subspaces (default: 60)",
    )

    # Monitoring and checkpointing
    mon = parser.add_argument_group("monitoring")
    mon.add_argument(
        "--audit-interval",
        type=int,
        default=500,
        help="Full SVD subspace audit every N steps (default: 500)",
    )
    mon.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=None,
        help="Explicit checkpoint steps for post-hoc auditing",
    )

    args = parser.parse_args()

    return train_geometric(
        size=args.size,
        seed=args.seed,
        device_str=args.device,
        output_dir=args.output,
        dataset=args.dataset,
        geo_weight=args.geo_weight,
        geo_interval=args.geo_interval,
        geo_pairs=args.geo_pairs,
        angle_floor=args.angle_floor,
        audit_interval=args.audit_interval,
        checkpoint_steps=args.checkpoint_steps,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
