#!/usr/bin/env python3
"""Extract behavioral subspace geometry around the overrefusal phase transition.

Runs subspace extraction at 9 checkpoints of the 7M hires model:
  - Pre-transition baselines: steps 800, 1000
  - Transition zone: steps 1700, 1800, 1900, 2000, 2100, 2200
  - Post-transition: step 2800

Extracts subspaces for all contrast-pair-compatible behaviors:
  factual, sycophancy, toxicity, bias, refusal, deception

Output: subspace_report.json in each checkpoint dir + summary table.

Usage:
    /Users/bryan/miniconda3/bin/python experiments/scale_ladder/extract_transition_geometry.py
    /Users/bryan/miniconda3/bin/python experiments/scale_ladder/extract_transition_geometry.py --force
"""

import argparse
import json
import sys
import re
import gc
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale_ladder.scale_audit import load_checkpoint

# -- Configuration --------------------------------------------------------

STEPS = [800, 1000, 1700, 1800, 1900, 2000, 2100, 2200, 2800]

BEHAVIORS = ["factual", "sycophancy", "toxicity", "bias", "refusal", "deception"]

BASE_DIR = Path("/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/results/scale_ladder/7M_seed42_hires")

DEVICE = "cpu"


def _patched_load_probes(behavior):
    """Load probes using v2 behavior plugin system (supports refusal, deception)."""
    if behavior == "factual":
        from rho_eval.probes import get_all_probes
        return get_all_probes()
    else:
        try:
            from rho_eval.behaviors import get_behavior
            beh = get_behavior(behavior)
            return beh.load_probes(seed=42)
        except Exception:
            from rho_eval.behavioral import load_behavioral_probes
            return load_behavioral_probes(behavior, seed=42)


def parse_overlap_matrix(s):
    """Parse the string representation of OverlapMatrix."""
    beh_match = re.search(r"behaviors=\[([^\]]+)\]", s)
    if not beh_match:
        return None, None, None, None
    behaviors = [b.strip().strip("'\"" ) for b in beh_match.group(1).split(",")]

    angles_match = re.search(r"subspace_angles=(\[\[.*?\]\])", s)
    cosine_match = re.search(r"cosine_matrix=(\[\[.*?\]\])", s)
    shared_match = re.search(r"shared_variance=(\[\[.*?\]\])", s)

    angles = json.loads(angles_match.group(1)) if angles_match else None
    cosines = json.loads(cosine_match.group(1)) if cosine_match else None
    shared = json.loads(shared_match.group(1)) if shared_match else None

    return behaviors, angles, cosines, shared


def run_extraction(model, tokenizer, device="cpu"):
    """Run subspace extraction with all 6 behaviors directly."""
    from rho_eval.interpretability.subspaces import extract_subspaces
    from rho_eval.interpretability.overlap import compute_overlap

    n_layers = model.config.n_layer
    all_layers = list(range(n_layers))

    print()
    print("  Extracting subspaces at all {} layers...".format(n_layers), flush=True)
    print("  Behaviors:", BEHAVIORS)
    t0 = time.time()

    subspaces = extract_subspaces(
        model, tokenizer,
        behaviors=BEHAVIORS,
        layers=all_layers,
        device=device,
        max_rank=50,
        verbose=True,
    )

    elapsed = time.time() - t0
    print("  Subspace extraction complete in {:.0f}s".format(elapsed), flush=True)

    print()
    print("  Computing pairwise overlaps (Grassmann angles)...", flush=True)
    overlap = compute_overlap(subspaces, top_k=10)

    eff_dim_data = {}
    for beh_name, layer_dict in subspaces.items():
        eff_dim_data[beh_name] = {}
        for layer_idx, result in layer_dict.items():
            ed = result.effective_dim
            ev = result.explained_variance
            eff_dim_data[beh_name][layer_idx] = {
                "effective_dim": ed,
                "explained_variance_90": float(ev[min(ed, len(ev) - 1)]) if ed < len(ev) else 1.0,
                "total_variance_captured": float(ev[-1]) if len(ev) > 0 else 0.0,
                "singular_values_top5": [float(v) for v in result.singular_values[:5]],
            }

    return subspaces, overlap, eff_dim_data


def validate_report(report_path):
    """Check if a cached report actually has all 6 behaviors in the overlap data."""
    try:
        data = json.loads(report_path.read_text())
        overlap = data.get("overlap", {})
        for key, val in overlap.items():
            if isinstance(val, str) and "refusal" in val:
                return True
        return False
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract transition geometry")
    parser.add_argument("--force", action="store_true",
                        help="Force re-extraction even if cached reports exist")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  SUBSPACE EXTRACTION: Overrefusal Phase Transition Geometry")
    print("  Steps:", STEPS)
    print("  Behaviors:", BEHAVIORS)
    print("  Device:", DEVICE)
    if args.force:
        print("  Mode: FORCE (re-extracting all)")
    print("=" * 70)
    print()

    t_total = time.time()

    # Verify all checkpoint dirs exist
    missing = []
    for step in STEPS:
        ckpt = BASE_DIR / "checkpoint_{}".format(step)
        if not ckpt.exists():
            missing.append(step)
    if missing:
        print("  ERROR: Missing checkpoints for steps:", missing)
        return 1

    # Monkey-patch the probe loader to support refusal/deception
    import rho_eval.interpretability.subspaces as submod
    submod._load_probes_for_behavior = _patched_load_probes
    print("  Patched _load_probes_for_behavior for v2 behavior support")

    results_by_step = {}

    for i, step in enumerate(STEPS):
        ckpt_dir = BASE_DIR / "checkpoint_{}".format(step)
        report_path = ckpt_dir / "subspace_report.json"

        if not args.force and report_path.exists() and validate_report(report_path):
            tag = "[{}/{}]".format(i + 1, len(STEPS))
            print("  {} Step {}: validated cache (has refusal in overlap), loading...".format(tag, step))
            results_by_step[step] = json.loads(report_path.read_text())
            continue

        sep = chr(0x2500) * 50
        tag = "[{}/{}]".format(i + 1, len(STEPS))
        print()
        print("  {} Step {} {}".format(tag, step, sep))
        t0 = time.time()

        model, tokenizer = load_checkpoint(str(ckpt_dir), device=DEVICE)

        subspaces, overlap, eff_dim_data = run_extraction(model, tokenizer, device=DEVICE)

        overlap_serializable = {}
        for key, matrix in overlap.items():
            if hasattr(matrix, "tolist"):
                overlap_serializable[key] = matrix.tolist()
            elif isinstance(matrix, dict):
                overlap_serializable[key] = {
                    str(k): v.tolist() if hasattr(v, "tolist") else v
                    for k, v in matrix.items()
                }
            else:
                overlap_serializable[key] = str(matrix)

        subspace_data = {
            "checkpoint": str(ckpt_dir),
            "step": step,
            "behaviors": BEHAVIORS,
            "n_layers": model.config.n_layer,
            "effective_dimensionality": {
                beh: {str(k): v for k, v in layers.items()}
                for beh, layers in eff_dim_data.items()
            },
            "overlap": overlap_serializable,
        }

        report_path.write_text(json.dumps(subspace_data, indent=2, default=str))
        print("  Saved:", report_path)

        results_by_step[step] = subspace_data

        elapsed = time.time() - t0
        print("  Step {} extracted in {:.0f}s ({:.1f} min)".format(step, elapsed, elapsed / 60))

        del model, subspaces, overlap
        gc.collect()

    # -- Parse Grassmann angles across steps --
    print()
    print("=" * 70)
    print("  GRASSMANN ANGLE TRAJECTORIES")
    print("=" * 70)
    print()

    pair_trajectories = {}

    for step in STEPS:
        data = results_by_step.get(step)
        if data is None:
            continue

        overlap = data.get("overlap", {})
        pair_angles_sum = {}
        pair_cosines_sum = {}
        pair_shared_sum = {}
        pair_count = {}

        for layer_key, overlap_str in overlap.items():
            if not isinstance(overlap_str, str):
                continue
            behaviors, angles, cosines, shared = parse_overlap_matrix(overlap_str)
            if behaviors is None or angles is None:
                continue
            n = len(behaviors)
            for ii in range(n):
                for jj in range(ii + 1, n):
                    pair = (behaviors[ii], behaviors[jj])
                    if pair not in pair_angles_sum:
                        pair_angles_sum[pair] = 0.0
                        pair_cosines_sum[pair] = 0.0
                        pair_shared_sum[pair] = 0.0
                        pair_count[pair] = 0
                    pair_angles_sum[pair] += angles[ii][jj]
                    pair_cosines_sum[pair] += abs(cosines[ii][jj])
                    pair_shared_sum[pair] += shared[ii][jj]
                    pair_count[pair] += 1

        for pair in pair_angles_sum:
            if pair not in pair_trajectories:
                pair_trajectories[pair] = []
            c = pair_count[pair]
            pair_trajectories[pair].append({
                "step": step,
                "mean_angle": round(pair_angles_sum[pair] / c, 2),
                "mean_abs_cosine": round(pair_cosines_sum[pair] / c, 4),
                "mean_shared_var": round(pair_shared_sum[pair] / c, 4),
            })

    key_pairs = [
        ("factual", "refusal"),
        ("factual", "toxicity"),
        ("factual", "bias"),
        ("factual", "sycophancy"),
        ("factual", "deception"),
        ("refusal", "toxicity"),
        ("refusal", "deception"),
        ("refusal", "bias"),
        ("bias", "toxicity"),
        ("sycophancy", "refusal"),
        ("deception", "toxicity"),
    ]

    for pair in key_pairs:
        traj = pair_trajectories.get(pair)
        if traj is None:
            traj = pair_trajectories.get((pair[1], pair[0]))
        if traj is None:
            continue

        print()
        print("  {} <-> {}:".format(pair[0], pair[1]))
        print("  {:>6s} | {:>12s} | {:>8s} | {:>10s}".format(
            "Step", "Angle (deg)", "|cos|", "SharedVar"))
        hsep = chr(0x2500) * 45
        print("  {}".format(hsep))
        for entry in traj:
            marker = ""
            if entry["step"] == 1900:
                marker = "  <- pre-transition"
            elif entry["step"] == 2000:
                marker = "  <- POST-TRANSITION"
            print(
                "  {:6d} | {:12.2f} | {:8.4f} | {:10.4f}{}".format(
                    entry["step"],
                    entry["mean_angle"],
                    entry["mean_abs_cosine"],
                    entry["mean_shared_var"],
                    marker,
                )
            )

    # Per-layer detail for factual<->refusal
    print()
    print()
    print("  PER-LAYER factual<->refusal at all steps:")
    print("  {:>6s} | {:>8s} | {:>8s} | {:>8s} | {:>8s}".format(
        "Step", "L0", "L1", "L2", "L3"))
    hsep = chr(0x2500) * 50
    print("  {}".format(hsep))

    for step in STEPS:
        data = results_by_step.get(step)
        if data is None:
            continue
        overlap = data.get("overlap", {})
        angles_per_layer = []
        for layer_idx in range(4):
            overlap_str = overlap.get(str(layer_idx), "")
            if not isinstance(overlap_str, str):
                angles_per_layer.append("N/A")
                continue
            behaviors, angles, _, _ = parse_overlap_matrix(overlap_str)
            if behaviors is None or angles is None:
                angles_per_layer.append("N/A")
                continue
            try:
                fi = behaviors.index("factual")
                ri = behaviors.index("refusal")
                angles_per_layer.append("{:.2f}".format(angles[fi][ri]))
            except ValueError:
                angles_per_layer.append("N/A")

        while len(angles_per_layer) < 4:
            angles_per_layer.append("N/A")

        marker = ""
        if step == 1900:
            marker = " <- pre"
        elif step == 2000:
            marker = " <- POST"
        print("  {:6d} | {:>8s} | {:>8s} | {:>8s} | {:>8s}{}".format(
            step,
            angles_per_layer[0],
            angles_per_layer[1],
            angles_per_layer[2],
            angles_per_layer[3],
            marker,
        ))

    # Save summary
    summary = {
        "experiment": "overrefusal_phase_transition_geometry",
        "steps": STEPS,
        "behaviors": BEHAVIORS,
        "pair_trajectories": {
            "{}_x_{}".format(p[0], p[1]): traj
            for p, traj in pair_trajectories.items()
        },
    }
    summary_path = BASE_DIR / "transition_geometry_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print()
    print("  Summary saved:", summary_path)

    total_elapsed = time.time() - t_total
    print()
    print("=" * 70)
    print("  COMPLETE: {} checkpoints, {:.1f} min total".format(len(STEPS), total_elapsed / 60))
    print("=" * 70)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
