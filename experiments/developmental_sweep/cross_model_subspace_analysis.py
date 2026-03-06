#!/usr/bin/env python3
"""Cross-model subspace alignment analysis: bias-only vs syco-only injection.

Tests whether two injection types activate the same mechanism or different
mechanisms that happen to produce similar outputs. Runs 4-part analysis:
  1. Per-probe behavioral sanity checks (pandas)
  2. Mean activation-difference vector alignment (cosine)
  3. Top singular vector / principal angle (SVD per layer)
  4. Full subspace overlap (Grassmann distance if eff_dim > 1)

Usage:
    python experiments/developmental_sweep/cross_model_subspace_analysis.py \
        --scale 5M --device mps

    # Both scales
    python experiments/developmental_sweep/cross_model_subspace_analysis.py \
        --scale 5M 7M --device mps
"""

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale_ladder.scale_audit import load_checkpoint
from rho_eval.interpretability.activation import (
    LayerActivationCapture,
    build_contrast_pairs,
)
from rho_eval.interpretability.subspaces import _load_probes_for_behavior


# ── Helpers (from overlap.py, inlined for clarity) ──────────────────────

def cosine_sim(v1, v2):
    """Cosine similarity between two 1D tensors."""
    v1f, v2f = v1.float(), v2.float()
    n1, n2 = v1f.norm(), v2f.norm()
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(torch.dot(v1f, v2f) / (n1 * n2))


def shared_variance(V1, V2):
    """Grassmann overlap: ||V1^T @ V2||_F / sqrt(k)."""
    V1f, V2f = V1.float(), V2.float()
    k = min(V1f.shape[0], V2f.shape[0])
    if k == 0:
        return 0.0
    C = V1f[:k] @ V2f[:k].T
    return float(C.norm() / math.sqrt(k))


def principal_angles_deg(V1, V2):
    """Principal angles between two subspaces, returned in degrees."""
    V1f, V2f = V1.float(), V2.float()
    k = min(V1f.shape[0], V2f.shape[0])
    if k == 0:
        return [90.0]
    C = V1f[:k] @ V2f[:k].T
    S = torch.linalg.svdvals(C).clamp(-1.0, 1.0)
    return [float(a) * 180.0 / math.pi for a in torch.acos(S)]


def effective_dim(S_vals, threshold=0.90):
    """Effective dimensionality: min k where cumvar >= threshold."""
    s2 = S_vals ** 2
    total = s2.sum()
    if total < 1e-12:
        return len(S_vals)
    cumvar = s2.cumsum(0) / total
    for i, cv in enumerate(cumvar):
        if cv >= threshold:
            return i + 1
    return len(S_vals)


# ── Part 1: Per-probe behavioral sanity checks ─────────────────────────

def part1_behavioral(details_van, details_bia, details_syc):
    """Per-probe behavioral sanity checks on shared parsed set.

    Returns dict with:
      - shared_probe_ids: list of IDs
      - n_shared: count
      - bias_only stats: accuracy, biased_rate, neither_rate
      - syco_only stats: accuracy, biased_rate, neither_rate
      - agreement: fraction of shared probes where both pick same answer
      - pearson_is_correct: correlation of binary correctness vectors
      - per_category: breakdown by BBQ category
    """
    # Index by probe ID
    bia_by_id = {d["id"]: d for d in details_bia}
    syc_by_id = {d["id"]: d for d in details_syc}
    van_by_id = {d["id"]: d for d in details_van}

    # Shared parsed set: probes where BOTH injection models parsed (model_answer not null)
    shared_ids = sorted(
        pid for pid in bia_by_id
        if pid in syc_by_id
        and bia_by_id[pid]["model_answer"] is not None
        and syc_by_id[pid]["model_answer"] is not None
    )

    n = len(shared_ids)
    if n == 0:
        return {"n_shared": 0, "error": "No shared parsed probes"}

    # Compute per-model stats on shared set
    def stats(by_id, ids):
        correct = sum(1 for pid in ids if by_id[pid]["is_correct"])
        biased = sum(1 for pid in ids if by_id[pid]["is_biased"])
        neither = sum(
            1 for pid in ids
            if not by_id[pid]["is_correct"] and not by_id[pid]["is_biased"]
        )
        return {
            "accuracy": correct / len(ids),
            "biased_rate": biased / len(ids),
            "neither_rate": neither / len(ids),
            "correct": correct,
            "biased": biased,
            "neither": neither,
        }

    bia_stats = stats(bia_by_id, shared_ids)
    syc_stats = stats(syc_by_id, shared_ids)

    # Per-probe agreement: same model_answer
    same_answer = sum(
        1 for pid in shared_ids
        if bia_by_id[pid]["model_answer"] == syc_by_id[pid]["model_answer"]
    )

    # Pearson correlation of binary is_correct vectors
    bia_correct = np.array([int(bia_by_id[pid]["is_correct"]) for pid in shared_ids])
    syc_correct = np.array([int(syc_by_id[pid]["is_correct"]) for pid in shared_ids])

    # Pearson: handle edge case where one vector is constant
    if bia_correct.std() < 1e-10 or syc_correct.std() < 1e-10:
        pearson_r = 0.0
    else:
        pearson_r = float(np.corrcoef(bia_correct, syc_correct)[0, 1])

    # McNemar: probes correct in one but not the other
    both_correct = sum(
        1 for pid in shared_ids
        if bia_by_id[pid]["is_correct"] and syc_by_id[pid]["is_correct"]
    )
    bia_only_correct = sum(
        1 for pid in shared_ids
        if bia_by_id[pid]["is_correct"] and not syc_by_id[pid]["is_correct"]
    )
    syc_only_correct = sum(
        1 for pid in shared_ids
        if not bia_by_id[pid]["is_correct"] and syc_by_id[pid]["is_correct"]
    )
    neither_correct = n - both_correct - bia_only_correct - syc_only_correct

    # Per-category breakdown
    cats = sorted(set(bia_by_id[pid]["category"] for pid in shared_ids))
    per_category = {}
    for cat in cats:
        cat_ids = [pid for pid in shared_ids if bia_by_id[pid]["category"] == cat]
        per_category[cat] = {
            "n": len(cat_ids),
            "bia_accuracy": sum(1 for pid in cat_ids if bia_by_id[pid]["is_correct"]) / len(cat_ids),
            "syc_accuracy": sum(1 for pid in cat_ids if syc_by_id[pid]["is_correct"]) / len(cat_ids),
            "agreement": sum(
                1 for pid in cat_ids
                if bia_by_id[pid]["model_answer"] == syc_by_id[pid]["model_answer"]
            ) / len(cat_ids),
        }

    return {
        "n_shared": n,
        "n_bia_parsed": sum(1 for d in details_bia if d["model_answer"] is not None),
        "n_syc_parsed": sum(1 for d in details_syc if d["model_answer"] is not None),
        "n_total": len(details_bia),
        "jaccard": n / len(
            set(pid for pid in bia_by_id if bia_by_id[pid]["model_answer"] is not None)
            | set(pid for pid in syc_by_id if syc_by_id[pid]["model_answer"] is not None)
        ),
        "bias_only_stats": bia_stats,
        "syco_only_stats": syc_stats,
        "same_answer_rate": same_answer / n,
        "same_answer_count": same_answer,
        "pearson_is_correct": round(pearson_r, 4),
        "mcnemar": {
            "both_correct": both_correct,
            "bia_only_correct": bia_only_correct,
            "syc_only_correct": syc_only_correct,
            "neither_correct": neither_correct,
        },
        "per_category": per_category,
        "shared_probe_ids": shared_ids,
    }


# ── Parts 2-4: Activation extraction + analysis ────────────────────────

@torch.no_grad()
def extract_activations_for_probes(
    model, tokenizer, probes, probe_ids, layers, device="cpu"
):
    """Extract per-probe activation-difference vectors at specified layers.

    Uses standard bias contrast pairs (text + correct vs text + biased)
    from shipped probe data.

    Returns:
        dict[layer_idx -> tensor of shape (n_probes, hidden_dim)]
    """
    # Build contrast pairs from full probe set, then filter to shared IDs
    pairs = build_contrast_pairs("bias", probes)
    pair_by_id = {p["id"]: p for p in pairs}

    # Filter to shared probe IDs (maintain order)
    selected_pairs = [pair_by_id[pid] for pid in probe_ids if pid in pair_by_id]

    if len(selected_pairs) < len(probe_ids):
        missing = len(probe_ids) - len(selected_pairs)
        print(f"    WARNING: {missing} probe IDs not found in contrast pairs")

    cap = LayerActivationCapture(model, layers)
    diffs_by_layer = {l: [] for l in layers}

    for i, pair in enumerate(selected_pairs):
        # Positive (correct answer)
        inputs = tokenizer(
            pair["positive"], return_tensors="pt",
            truncation=True, max_length=512,
        ).to(device)
        model(**inputs)
        pos_h = {l: cap.get(l)[0, -1, :].cpu() for l in layers}
        cap.clear()

        # Negative (biased answer)
        inputs = tokenizer(
            pair["negative"], return_tensors="pt",
            truncation=True, max_length=512,
        ).to(device)
        model(**inputs)
        neg_h = {l: cap.get(l)[0, -1, :].cpu() for l in layers}
        cap.clear()

        for l in layers:
            diffs_by_layer[l].append(pos_h[l] - neg_h[l])

        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(selected_pairs)} probes", flush=True)

    cap.remove()

    return {l: torch.stack(vecs) for l, vecs in diffs_by_layer.items()}


def part2_mean_vector(diffs_van, diffs_bia, diffs_syc, layers):
    """Mean activation-difference vector alignment."""
    results = {}
    for l in layers:
        mu_van = diffs_van[l].mean(dim=0)
        mu_bia = diffs_bia[l].mean(dim=0)
        mu_syc = diffs_syc[l].mean(dim=0)

        results[str(l)] = {
            "bia_vs_syc": round(cosine_sim(mu_bia, mu_syc), 4),
            "bia_vs_van": round(cosine_sim(mu_bia, mu_van), 4),
            "syc_vs_van": round(cosine_sim(mu_syc, mu_van), 4),
            "mu_norm_van": round(float(mu_van.norm()), 4),
            "mu_norm_bia": round(float(mu_bia.norm()), 4),
            "mu_norm_syc": round(float(mu_syc.norm()), 4),
        }
    return results


def part3_top_singular(diffs_van, diffs_bia, diffs_syc, layers):
    """Top singular vector / principal angle analysis."""
    results = {}
    svd_data = {}  # Save for Part 4

    for l in layers:
        layer_svd = {}
        for name, D in [("van", diffs_van[l]), ("bia", diffs_bia[l]), ("syc", diffs_syc[l])]:
            D_centered = D - D.mean(dim=0)
            U, S, Vh = torch.linalg.svd(D_centered.float(), full_matrices=False)
            layer_svd[name] = {"S": S, "Vh": Vh}

        svd_data[l] = layer_svd

        # Top-1 cosines
        vh1_bia = layer_svd["bia"]["Vh"][0]
        vh1_syc = layer_svd["syc"]["Vh"][0]
        vh1_van = layer_svd["van"]["Vh"][0]

        # Grassmann top-1 angle
        cos_bs = abs(cosine_sim(vh1_bia, vh1_syc))  # abs because sign is arbitrary for SVD
        angle_bs = math.degrees(math.acos(min(cos_bs, 1.0)))

        cos_bv = abs(cosine_sim(vh1_bia, vh1_van))
        angle_bv = math.degrees(math.acos(min(cos_bv, 1.0)))

        cos_sv = abs(cosine_sim(vh1_syc, vh1_van))
        angle_sv = math.degrees(math.acos(min(cos_sv, 1.0)))

        # Top-1 explained variance
        s_bia = layer_svd["bia"]["S"]
        s_syc = layer_svd["syc"]["S"]
        s_van = layer_svd["van"]["S"]

        def top1_var(S):
            total = (S ** 2).sum()
            return float(S[0] ** 2 / total) if total > 0 else 0.0

        results[str(l)] = {
            "bia_vs_syc_cosine": round(cos_bs, 4),
            "bia_vs_syc_angle_deg": round(angle_bs, 2),
            "bia_vs_van_cosine": round(cos_bv, 4),
            "bia_vs_van_angle_deg": round(angle_bv, 2),
            "syc_vs_van_cosine": round(cos_sv, 4),
            "syc_vs_van_angle_deg": round(angle_sv, 2),
            "top1_var_bia": round(top1_var(s_bia), 4),
            "top1_var_syc": round(top1_var(s_syc), 4),
            "top1_var_van": round(top1_var(s_van), 4),
        }

    return results, svd_data


def part4_full_subspace(svd_data, layers):
    """Full subspace overlap analysis (if eff_dim > 1)."""
    results = {}

    for l in layers:
        s_bia = svd_data[l]["bia"]["S"]
        s_syc = svd_data[l]["syc"]["S"]
        s_van = svd_data[l]["van"]["S"]

        ed_bia = effective_dim(s_bia, 0.90)
        ed_syc = effective_dim(s_syc, 0.90)
        ed_van = effective_dim(s_van, 0.90)

        # Use max effective dim as subspace rank for overlap
        k = max(ed_bia, ed_syc, 3)  # At least 3 for meaningful overlap

        Vh_bia = svd_data[l]["bia"]["Vh"][:k]
        Vh_syc = svd_data[l]["syc"]["Vh"][:k]
        Vh_van = svd_data[l]["van"]["Vh"][:k]

        # Shared variance (Grassmann overlap)
        sv_bs = shared_variance(Vh_bia, Vh_syc)
        sv_bv = shared_variance(Vh_bia, Vh_van)
        sv_sv = shared_variance(Vh_syc, Vh_van)

        # Principal angles (all k)
        angles_bs = principal_angles_deg(Vh_bia, Vh_syc)
        angles_bv = principal_angles_deg(Vh_bia, Vh_van)
        angles_sv = principal_angles_deg(Vh_syc, Vh_van)

        # Sum of cosines (alternative Grassmann metric)
        def sum_cos(angles):
            return sum(math.cos(math.radians(a)) for a in angles)

        # Per-probe cosine (activation-level, not subspace)
        # Already done in Part 2 via mean vectors; skip redundant computation

        results[str(l)] = {
            "eff_dim_bia": ed_bia,
            "eff_dim_syc": ed_syc,
            "eff_dim_van": ed_van,
            "rank_used": k,
            "bia_vs_syc": {
                "shared_variance": round(sv_bs, 4),
                "mean_principal_angle_deg": round(np.mean(angles_bs), 2),
                "principal_angles_deg": [round(a, 2) for a in angles_bs],
                "sum_of_cosines": round(sum_cos(angles_bs), 4),
            },
            "bia_vs_van": {
                "shared_variance": round(sv_bv, 4),
                "mean_principal_angle_deg": round(np.mean(angles_bv), 2),
                "principal_angles_deg": [round(a, 2) for a in angles_bv],
                "sum_of_cosines": round(sum_cos(angles_bv), 4),
            },
            "syc_vs_van": {
                "shared_variance": round(sv_sv, 4),
                "mean_principal_angle_deg": round(np.mean(angles_sv), 2),
                "principal_angles_deg": [round(a, 2) for a in angles_sv],
                "sum_of_cosines": round(sum_cos(angles_sv), 4),
            },
        }

    return results


# ── Summary printer ─────────────────────────────────────────────────────

def print_summary(output, scale):
    """Print human-readable summary."""
    p1 = output["part1_behavioral"]
    p2 = output["part2_mean_vector"]
    p3 = output["part3_top_singular"]
    p4 = output["part4_full_subspace"]
    layers = output["layers"]

    print(f"\n{'='*70}")
    print(f"  CROSS-MODEL SUBSPACE ANALYSIS — {scale}")
    print(f"{'='*70}")

    # Part 1
    print(f"\n  Part 1: Per-Probe Behavioral Sanity Checks")
    print(f"  {'─'*50}")
    print(f"  Shared parsed set: {p1['n_shared']} probes "
          f"(bia={p1['n_bia_parsed']}, syc={p1['n_syc_parsed']}, "
          f"Jaccard={p1['jaccard']:.3f})")
    print(f"\n  {'Model':<12s}  {'Accuracy':>8s}  {'Biased%':>8s}  {'Neither%':>8s}")
    print(f"  {'─'*40}")
    for label, st in [("bias-only", p1["bias_only_stats"]), ("syco-only", p1["syco_only_stats"])]:
        print(f"  {label:<12s}  {st['accuracy']:>7.1%}  {st['biased_rate']:>7.1%}  {st['neither_rate']:>7.1%}")
    print(f"\n  Same-answer rate: {p1['same_answer_rate']:.1%} ({p1['same_answer_count']}/{p1['n_shared']})")
    print(f"  Pearson(is_correct): {p1['pearson_is_correct']:.4f}")
    mc = p1["mcnemar"]
    print(f"  McNemar: both={mc['both_correct']}, bia_only={mc['bia_only_correct']}, "
          f"syc_only={mc['syc_only_correct']}, neither={mc['neither_correct']}")

    # Part 2
    print(f"\n  Part 2: Mean Activation-Difference Vector Alignment")
    print(f"  {'─'*50}")
    print(f"  {'Layer':>5s}  {'bia↔syc':>8s}  {'bia↔van':>8s}  {'syc↔van':>8s}  "
          f"{'‖μ‖ bia':>8s}  {'‖μ‖ syc':>8s}  {'‖μ‖ van':>8s}")
    for l in layers:
        d = p2[str(l)]
        print(f"  {l:>5d}  {d['bia_vs_syc']:>+8.4f}  {d['bia_vs_van']:>+8.4f}  "
              f"{d['syc_vs_van']:>+8.4f}  {d['mu_norm_bia']:>8.2f}  "
              f"{d['mu_norm_syc']:>8.2f}  {d['mu_norm_van']:>8.2f}")

    # Part 3
    print(f"\n  Part 3: Top Singular Vector / Principal Angle")
    print(f"  {'─'*50}")
    print(f"  {'Layer':>5s}  {'cos(bia↔syc)':>12s}  {'∠bia↔syc':>9s}  "
          f"{'top1%bia':>9s}  {'top1%syc':>9s}  {'top1%van':>9s}")
    for l in layers:
        d = p3[str(l)]
        print(f"  {l:>5d}  {d['bia_vs_syc_cosine']:>12.4f}  "
              f"{d['bia_vs_syc_angle_deg']:>8.1f}°  "
              f"{d['top1_var_bia']:>8.1%}  {d['top1_var_syc']:>8.1%}  "
              f"{d['top1_var_van']:>8.1%}")

    # Part 4
    print(f"\n  Part 4: Full Subspace Overlap")
    print(f"  {'─'*50}")
    for l in layers:
        d = p4[str(l)]
        print(f"  Layer {l}: eff_dim bia={d['eff_dim_bia']}, "
              f"syc={d['eff_dim_syc']}, van={d['eff_dim_van']} (rank={d['rank_used']})")
        bs = d["bia_vs_syc"]
        print(f"    bia↔syc: shared_var={bs['shared_variance']:.4f}, "
              f"mean∠={bs['mean_principal_angle_deg']:.1f}°, "
              f"Σcos={bs['sum_of_cosines']:.4f}")
        bv = d["bia_vs_van"]
        print(f"    bia↔van: shared_var={bv['shared_variance']:.4f}, "
              f"mean∠={bv['mean_principal_angle_deg']:.1f}°, "
              f"Σcos={bv['sum_of_cosines']:.4f}")
        sv = d["syc_vs_van"]
        print(f"    syc↔van: shared_var={sv['shared_variance']:.4f}, "
              f"mean∠={sv['mean_principal_angle_deg']:.1f}°, "
              f"Σcos={sv['sum_of_cosines']:.4f}")

    print()


# ── Main ────────────────────────────────────────────────────────────────

def run_analysis(scale, device="cpu"):
    """Run full 4-part analysis for one scale."""
    base = Path("results/scale_ladder")
    models = {
        "van": base / f"{scale}_seed42",
        "bia": base / f"{scale}_seed42_contr_bia_r20",
        "syc": base / f"{scale}_seed42_contr_syc_r20",
    }

    # Verify all exist
    for label, p in models.items():
        if not (p / "audit_details.json").exists():
            print(f"  ERROR: Missing audit_details.json for {label}: {p}")
            return None

    t0 = time.time()

    # ── Load per-probe details ──────────────────────────────────────
    print(f"\n  Loading audit details for {scale}...")
    details = {}
    for label, p in models.items():
        with open(p / "audit_details.json") as f:
            data = json.load(f)
        details[label] = data["behaviors"]["bias"]["details"]
        print(f"    {label}: {len(details[label])} bias probes, "
              f"d_model={data['d_model']}, n_layers={data['n_layers']}")

    # ── Part 1 ──────────────────────────────────────────────────────
    print(f"\n  Part 1: Per-probe behavioral sanity checks...")
    p1 = part1_behavioral(details["van"], details["bia"], details["syc"])
    shared_ids = p1["shared_probe_ids"]
    print(f"    Shared parsed set: {p1['n_shared']} probes")

    if p1["n_shared"] < 10:
        print("    ERROR: Too few shared probes for analysis")
        return None

    # ── Load probes and models for activation extraction ────────────
    print(f"\n  Loading bias probes...")
    probes = _load_probes_for_behavior("bias")
    print(f"    {len(probes)} probes loaded")

    # Determine layers
    with open(models["van"] / "audit_details.json") as f:
        n_layers = json.load(f)["n_layers"]
    layers = list(range(n_layers))
    print(f"    Layers to analyze: {layers}")

    # Extract activations for all 3 models
    diffs = {}
    for label, ckpt_path in models.items():
        print(f"\n  Extracting activations: {label} ({ckpt_path.name})...")
        model, tokenizer = load_checkpoint(str(ckpt_path), device=device)
        model.eval()
        diffs[label] = extract_activations_for_probes(
            model, tokenizer, probes, shared_ids, layers, device=device,
        )
        del model  # Free memory
        torch.cuda.empty_cache() if "cuda" in device else None
        print(f"    Shape: {diffs[label][layers[0]].shape}")

    # ── Part 2 ──────────────────────────────────────────────────────
    print(f"\n  Part 2: Mean vector alignment...")
    p2 = part2_mean_vector(diffs["van"], diffs["bia"], diffs["syc"], layers)

    # ── Part 3 ──────────────────────────────────────────────────────
    print(f"\n  Part 3: Top singular vector / principal angle...")
    p3, svd_data = part3_top_singular(diffs["van"], diffs["bia"], diffs["syc"], layers)

    # ── Part 4 ──────────────────────────────────────────────────────
    print(f"\n  Part 4: Full subspace overlap...")
    p4 = part4_full_subspace(svd_data, layers)

    elapsed = time.time() - t0

    # ── Assemble output ─────────────────────────────────────────────
    # Remove non-serializable probe IDs from p1 (keep count)
    p1_clean = {k: v for k, v in p1.items() if k != "shared_probe_ids"}

    output = {
        "scale": scale,
        "layers": layers,
        "n_layers": n_layers,
        "shared_probes": p1["n_shared"],
        "elapsed_seconds": round(elapsed, 1),
        "part1_behavioral": p1_clean,
        "part2_mean_vector": p2,
        "part3_top_singular": p3,
        "part4_full_subspace": p4,
    }

    # Print summary
    print_summary(output, scale)

    # Save
    out_dir = Path("results/developmental_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cross_model_subspace_{scale}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  Saved: {out_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model subspace alignment analysis"
    )
    parser.add_argument(
        "--scale", nargs="+", default=["5M"],
        help="Scale(s) to analyze (e.g., 5M 7M)"
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    for scale in args.scale:
        run_analysis(scale, device=args.device)

    print("\n  All done!")


if __name__ == "__main__":
    main()
