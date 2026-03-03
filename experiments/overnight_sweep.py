"""Overnight Sweep — gamma* dose-response + CatSAE experiment.

Orchestrates a full overnight GPU run:
  1. gamma* sweep: 3 surgery runs at gamma in {0.05, 0.10, 0.20}
     (validated: gamma=0.15 gave +12.6pp; confirm dose-response curve)
  2. CatSAE: category-aware SAE-informed surgery
     (hypothesis: vulnerability-weighted protection beats uniform)

Each surgery run:
  - Loads model fresh (MLX)
  - SVD compress (0.7)
  - LoRA SFT with rho=0.2 + gamma protection
  - Bias audit before/after
  - ~2.5h per run on 7B

Total estimated time: ~12-14h (4 surgery runs + CatSAE analysis).

Usage:
    # Full overnight run
    nohup python experiments/overnight_sweep.py Qwen/Qwen2.5-7B-Instruct &

    # gamma* sweep only (skip CatSAE)
    python experiments/overnight_sweep.py Qwen/Qwen2.5-7B-Instruct --gamma-only

    # CatSAE only (skip gamma* sweep)
    python experiments/overnight_sweep.py Qwen/Qwen2.5-7B-Instruct --catsae-only

    # With custom gamma values
    python experiments/overnight_sweep.py Qwen/Qwen2.5-7B-Instruct \
        --gammas 0.05 0.10 0.20 0.30
"""

import argparse
import gc
import json
import subprocess
import time
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # experiments/ for cat_sae_mlx


# ── Gamma* Sweep ───────────────────────────────────────────────────────


def run_gamma_surgery(
    model_name: str,
    gamma: float,
    output_dir: Path,
    protection_categories: list[str],
    compress_ratio: float = 0.7,
    rho_weight: float = 0.2,
) -> dict:
    """Run a single surgery iteration with explicit gamma value.

    Loads the model fresh each time (LoRA fusing mutates model).

    Returns:
        Result dict with baseline, final, config, timing.
    """
    import mlx_lm
    import mlx.core as mx

    run_dir = output_dir / f"gamma_{gamma:.2f}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}", flush=True)
    print(f"  GAMMA SWEEP: gamma={gamma}", flush=True)
    print(f"  Model:  {model_name}", flush=True)
    print(f"  Output: {run_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    t_start = time.time()

    # ── Load model ─────────────────────────────────────────────────
    print(f"  Loading model...", flush=True)
    t0 = time.time()
    model, tokenizer = mlx_lm.load(model_name)
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    # ── Baseline audit ─────────────────────────────────────────────
    from rho_eval.audit import audit

    print(f"\n  Baseline bias audit...", flush=True)
    t0 = time.time()
    report = audit(model=model, tokenizer=tokenizer, behaviors=["bias"], device="mlx")
    bias_result = report.behaviors["bias"]
    baseline = {
        "rho": bias_result.rho,
        "positive_count": bias_result.positive_count,
        "total": bias_result.total,
        "category_metrics": (bias_result.metadata or {}).get("category_metrics", {}),
        "elapsed": time.time() - t0,
    }
    print(f"  Baseline rho={baseline['rho']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # ── SVD compression ────────────────────────────────────────────
    if compress_ratio < 1.0:
        print(f"\n  SVD compression (ratio={compress_ratio})...", flush=True)
        t0 = time.time()
        import numpy as np

        compressed = 0
        for i, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            for proj_name in ["q_proj", "k_proj", "o_proj"]:
                if not hasattr(attn, proj_name):
                    continue
                proj = getattr(attn, proj_name)
                W_mx = proj.weight
                W = np.array(W_mx.astype(mx.float32))
                if len(W.shape) != 2 or min(W.shape) <= 10:
                    continue
                rank = max(1, int(min(W.shape) * compress_ratio))
                try:
                    U, S, Vh = np.linalg.svd(W, full_matrices=False)
                    W_approx = (U[:, :rank] * S[:rank]) @ Vh[:rank, :]
                    proj.weight = mx.array(W_approx).astype(W_mx.dtype)
                    compressed += 1
                except Exception:
                    pass

        mx.eval(model.parameters())
        print(f"  Compressed {compressed} matrices in {time.time()-t0:.1f}s",
              flush=True)

    # ── LoRA SFT with gamma protection ─────────────────────────────
    import random as _random
    from rho_eval.alignment.dataset import (
        BehavioralContrastDataset, _build_trap_texts, _load_alpaca_texts,
    )
    from rho_eval.alignment.mlx_trainer import mlx_rho_guided_sft

    # SFT data
    rng = _random.Random(42)
    sft_texts = []
    trap_texts = _build_trap_texts(["sycophancy"], seed=42)
    rng.shuffle(trap_texts)
    sft_texts.extend(trap_texts[:400])
    try:
        alpaca_texts = _load_alpaca_texts(1600, seed=42)
        sft_texts.extend(alpaca_texts)
    except Exception as e:
        print(f"  Alpaca load failed ({e}), using traps only", flush=True)
    rng.shuffle(sft_texts)
    sft_texts = sft_texts[:2000]

    # Contrast (sycophancy target)
    contrast_dataset = BehavioralContrastDataset(behaviors=["sycophancy"], seed=42)

    # Protection (bias, with category filter)
    protection_dataset = None
    if gamma > 0 and protection_categories:
        protection_dataset = BehavioralContrastDataset(
            behaviors=["bias"],
            categories=protection_categories,
            seed=42,
        )
        print(f"  Protection: {len(protection_dataset)} pairs "
              f"from {protection_categories}", flush=True)

    # Train
    print(f"\n  LoRA SFT: rho={rho_weight}, gamma={gamma}...", flush=True)
    t0 = time.time()
    sft_result = mlx_rho_guided_sft(
        model, tokenizer,
        sft_texts,
        contrast_dataset=contrast_dataset,
        rho_weight=rho_weight,
        gamma_weight=gamma,
        protection_dataset=protection_dataset,
        epochs=1,
        lr=2e-4,
        margin=0.1,
    )
    sft_elapsed = time.time() - t0
    print(f"  SFT complete: {sft_result['steps']} steps, {sft_elapsed:.0f}s",
          flush=True)

    # ── Final audit ────────────────────────────────────────────────
    print(f"\n  Final bias audit...", flush=True)
    t0 = time.time()
    report = audit(model=model, tokenizer=tokenizer, behaviors=["bias"], device="mlx")
    bias_result = report.behaviors["bias"]
    final = {
        "rho": bias_result.rho,
        "positive_count": bias_result.positive_count,
        "total": bias_result.total,
        "category_metrics": (bias_result.metadata or {}).get("category_metrics", {}),
        "elapsed": time.time() - t0,
    }
    print(f"  Final rho={final['rho']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    total_elapsed = time.time() - t_start
    delta = final["rho"] - baseline["rho"]

    print(f"\n  Result: {baseline['rho']:.4f} -> {final['rho']:.4f} "
          f"(delta={delta:+.4f})", flush=True)
    print(f"  Time: {total_elapsed/60:.1f} min", flush=True)

    # Per-category summary
    print(f"\n  {'Category':<30s} {'Before':>7s} {'After':>7s} {'Delta':>7s}",
          flush=True)
    print(f"  {'-'*54}", flush=True)
    for cat in sorted(final["category_metrics"].keys()):
        b = baseline["category_metrics"].get(cat, {}).get("accuracy", 0)
        a = final["category_metrics"][cat]["accuracy"]
        d = a - b
        marker = "FAIL" if d < -0.05 else ("+" if d > 0.05 else " ")
        print(f"  {cat:<30s} {b:>6.1%} {a:>6.1%} {d:>+6.1%} {marker}",
              flush=True)

    # Save result
    result = {
        "model": model_name,
        "config": {
            "gamma_weight": gamma,
            "rho_weight": rho_weight,
            "compress_ratio": compress_ratio,
            "protection_categories": protection_categories,
        },
        "baseline_bias": baseline,
        "final_bias": final,
        "sft_result": {k: v for k, v in sft_result.items() if k != "merged_model"},
        "total_elapsed_sec": round(total_elapsed, 1),
        "bias_delta": round(delta, 4),
    }

    result_path = run_dir / "surgery_result.json"
    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"  Saved to {result_path}", flush=True)

    # Cleanup
    del model, tokenizer
    gc.collect()

    return result


# ── Summary Table ──────────────────────────────────────────────────────


def print_summary(results: list[dict], catsae_result: dict | None = None):
    """Print a summary table of all sweep results."""
    print(f"\n{'='*70}", flush=True)
    print(f"  OVERNIGHT SWEEP SUMMARY", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Gamma sweep table
    if results:
        print(f"  Gamma Dose-Response:", flush=True)
        print(f"  {'Gamma':>6s} {'Baseline':>9s} {'Final':>9s} {'Delta':>9s} "
              f"{'Time':>7s}", flush=True)
        print(f"  {'-'*42}", flush=True)

        for r in sorted(results, key=lambda x: x["config"]["gamma_weight"]):
            g = r["config"]["gamma_weight"]
            b = r["baseline_bias"]["rho"]
            f = r["final_bias"]["rho"]
            d = r["bias_delta"]
            t = r["total_elapsed_sec"] / 60
            print(f"  {g:>6.2f} {b:>9.4f} {f:>9.4f} {d:>+9.4f} {t:>6.1f}m",
                  flush=True)

        # Previous validated gamma=0.15 result available in results/surgery/
        print(f"  (* see results/surgery/ for gamma=0.15 baseline)", flush=True)

    # Per-category comparison
    if len(results) >= 2:
        print(f"\n  Per-Category Results by Gamma:", flush=True)
        all_cats = set()
        for r in results:
            all_cats |= set(r["final_bias"].get("category_metrics", {}).keys())

        # Header
        gammas = sorted(set(r["config"]["gamma_weight"] for r in results))
        header = f"  {'Category':<25s}"
        for g in gammas:
            header += f" g={g:.2f}"
        print(header, flush=True)
        print(f"  {'-'*55}", flush=True)

        for cat in sorted(all_cats):
            line = f"  {cat:<25s}"
            for g in gammas:
                r = next((x for x in results if x["config"]["gamma_weight"] == g), None)
                if r:
                    before = r["baseline_bias"].get("category_metrics", {}).get(
                        cat, {}
                    ).get("accuracy", 0)
                    after = r["final_bias"].get("category_metrics", {}).get(
                        cat, {}
                    ).get("accuracy", 0)
                    delta = after - before
                    line += f" {delta:>+5.1%}"
                else:
                    line += "    --"
            print(line, flush=True)

    # CatSAE result
    if catsae_result:
        print(f"\n  CatSAE Result:", flush=True)
        surgery = catsae_result.get("surgery", {})
        b = surgery.get("baseline_bias", {}).get("rho", 0)
        f = surgery.get("final_bias", {}).get("rho", 0)
        d = f - b
        print(f"  Bias: {b:.4f} -> {f:.4f} (delta={d:+.4f})", flush=True)

        # Top vulnerable categories and their outcomes
        vuln = catsae_result.get("census", {}).get("vulnerability", {})
        weights = catsae_result.get("protection_weights", {})
        if vuln:
            print(f"\n  CatSAE Category Detail:", flush=True)
            print(f"  {'Category':<25s} {'Vuln':>5s} {'Weight':>6s} {'Delta':>7s}",
                  flush=True)
            print(f"  {'-'*48}", flush=True)
            baseline_cats = surgery.get("baseline_bias", {}).get("category_metrics", {})
            final_cats = surgery.get("final_bias", {}).get("category_metrics", {})
            for cat in sorted(vuln, key=lambda c: vuln[c].get("vulnerability_score", 0),
                              reverse=True):
                v = vuln[cat].get("vulnerability_score", 0)
                w = weights.get(cat, 0)
                b = baseline_cats.get(cat, {}).get("accuracy", 0)
                a = final_cats.get(cat, {}).get("accuracy", 0)
                d = a - b
                if v > 0.1 or w > 0:
                    print(f"  {cat:<25s} {v:>5.3f} {w:>5.2f}x {d:>+6.1%}",
                          flush=True)

    # Total time
    total_time = sum(r["total_elapsed_sec"] for r in results)
    if catsae_result:
        total_time += catsae_result.get("total_elapsed_sec", 0)
    print(f"\n  Total wall time: {total_time/3600:.1f}h", flush=True)
    print(f"{'='*70}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Overnight Sweep — gamma* + CatSAE experiments",
    )
    parser.add_argument("model", help="HuggingFace model ID")
    parser.add_argument(
        "--gammas", type=float, nargs="+",
        default=[0.05, 0.10, 0.20],
        help="Gamma values for dose-response sweep (default: 0.05 0.10 0.20)",
    )
    parser.add_argument(
        "--gamma-only", action="store_true",
        help="Run gamma* sweep only (skip CatSAE)",
    )
    parser.add_argument(
        "--catsae-only", action="store_true",
        help="Run CatSAE only (skip gamma* sweep)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="results/overnight_sweep",
    )
    args = parser.parse_args()

    model_short = args.model.split("/")[-1]
    output_dir = Path(args.output) / model_short
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    print(f"\n{'='*70}", flush=True)
    print(f"  OVERNIGHT SWEEP — {timestamp}", flush=True)
    print(f"  Model:  {args.model}", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    if not args.catsae_only:
        print(f"  Gammas: {args.gammas}", flush=True)
    print(f"  Mode:   {'gamma only' if args.gamma_only else 'catsae only' if args.catsae_only else 'full (gamma + catsae)'}", flush=True)
    n_runs = (0 if args.catsae_only else len(args.gammas)) + (0 if args.gamma_only else 1)
    est_hours = n_runs * 2.5
    print(f"  Est:    ~{est_hours:.0f}h ({n_runs} GPU runs)", flush=True)
    print(f"{'='*70}\n", flush=True)

    t_global = time.time()

    # Protection categories (same as validated surgery)
    protection_categories = [
        "Age", "Gender_biology", "Race_ethnicity", "Sexual_orientation_biology",
    ]

    # ── Gamma* Sweep ───────────────────────────────────────────────
    gamma_results = []
    if not args.catsae_only:
        for i, gamma in enumerate(args.gammas):
            print(f"\n  === Gamma Run {i+1}/{len(args.gammas)}: gamma={gamma} ===\n",
                  flush=True)
            try:
                result = run_gamma_surgery(
                    args.model, gamma, output_dir,
                    protection_categories=protection_categories,
                )
                gamma_results.append(result)
            except Exception as e:
                print(f"\n  ERROR in gamma={gamma}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                gamma_results.append({
                    "config": {"gamma_weight": gamma},
                    "baseline_bias": {"rho": 0},
                    "final_bias": {"rho": 0, "category_metrics": {}},
                    "bias_delta": 0,
                    "total_elapsed_sec": 0,
                    "error": str(e),
                })

            # Force cleanup between runs
            gc.collect()
            time.sleep(2)

    # ── CatSAE Experiment ──────────────────────────────────────────
    catsae_result = None
    if not args.gamma_only:
        print(f"\n  === CatSAE Experiment ===\n", flush=True)
        catsae_dir = output_dir / "cat_sae"
        catsae_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Import and run CatSAE inline (avoids subprocess issues with nohup)
            from cat_sae_mlx import (
                collect_mlx_activations,
                train_sae_on_activations,
                category_feature_census,
                build_weighted_protection,
                run_catsae_surgery,
            )
            import mlx_lm

            t_cat = time.time()

            # Phase 1: Load model + collect activations
            print(f"  CatSAE Phase 1: Loading model + collecting activations...",
                  flush=True)
            model, tokenizer = mlx_lm.load(args.model)
            act_data = collect_mlx_activations(model, tokenizer, max_probes_per_behavior=150)

            # Save activations
            import numpy as np
            np.savez_compressed(
                catsae_dir / "activations.npz",
                activations=act_data["activations"],
                labels=np.array(act_data["labels"]),
                polarities=np.array(act_data["polarities"]),
                categories=np.array(act_data["categories"]),
            )

            # Phase 2: Train SAE
            print(f"\n  CatSAE Phase 2: Training SAE...", flush=True)
            sae, sae_stats = train_sae_on_activations(act_data)
            sae.save(catsae_dir / "gated_sae.pt")

            # Phase 3: Feature census
            print(f"\n  CatSAE Phase 3: Feature census...", flush=True)
            census = category_feature_census(sae, act_data)

            # Save census
            census_save = {
                "behavior_features": {
                    b: {"n": len(f), "top_10": f[:10]}
                    for b, f in census["behavior_features"].items()
                },
                "category_features": {
                    c: {"n": len(f), "top_10": f[:10]}
                    for c, f in census["category_features"].items()
                },
                "sycophancy_exposure": census["sycophancy_exposure"],
                "vulnerability": census["vulnerability"],
                "n_sycophancy_features": census["n_sycophancy_features"],
                "n_bias_features": census["n_bias_features"],
            }
            (catsae_dir / "category_census.json").write_text(
                json.dumps(census_save, indent=2)
            )

            # Phase 4: Build weighted protection
            print(f"\n  CatSAE Phase 4: Building weighted protection...", flush=True)
            vulnerability = census["vulnerability"]
            known_protect = [
                "Age", "Gender_biology", "Race_ethnicity", "Sexual_orientation_biology",
            ]
            sae_protect = [
                cat for cat in vulnerability
                if vulnerability[cat]["vulnerability_score"] > 0.3
            ]
            cats_to_protect = sorted(set(known_protect) | set(sae_protect))

            protection_dataset, weights, cat_counts = build_weighted_protection(
                vulnerability, cats_to_protect,
            )

            # Phase 5: Surgery
            print(f"\n  CatSAE Phase 5: Surgery...", flush=True)
            surgery_result = run_catsae_surgery(
                model, tokenizer, args.model,
                protection_dataset=protection_dataset,
                gamma_weight=0.15,
            )

            catsae_elapsed = time.time() - t_cat

            catsae_result = {
                "model": args.model,
                "config": {
                    "gamma_weight": 0.15,
                    "method": "catsae_weighted",
                },
                "sae_stats": sae_stats,
                "census": census_save,
                "protection_weights": {c: round(w, 3) for c, w in weights.items()},
                "protection_pair_counts": cat_counts,
                "surgery": {
                    "baseline_bias": surgery_result["baseline"],
                    "final_bias": surgery_result["final"],
                    "sft_result": surgery_result["sft_result"],
                },
                "total_elapsed_sec": round(catsae_elapsed, 1),
            }

            (catsae_dir / "cat_sae_result.json").write_text(
                json.dumps(catsae_result, indent=2, default=str)
            )

            del model, tokenizer
            gc.collect()

        except Exception as e:
            print(f"\n  ERROR in CatSAE: {e}", flush=True)
            import traceback
            traceback.print_exc()

    # ── Summary ────────────────────────────────────────────────────
    print_summary(gamma_results, catsae_result)

    # Save combined manifest
    manifest = {
        "model": args.model,
        "timestamp": timestamp,
        "gammas_tested": args.gammas if not args.catsae_only else [],
        "gamma_results": [
            {
                "gamma": r["config"]["gamma_weight"],
                "baseline_rho": r["baseline_bias"]["rho"],
                "final_rho": r["final_bias"]["rho"],
                "delta": r["bias_delta"],
                "time_sec": r["total_elapsed_sec"],
                "error": r.get("error"),
            }
            for r in gamma_results
        ],
        "catsae_ran": catsae_result is not None,
        "total_wall_time_sec": round(time.time() - t_global, 1),
    }
    (output_dir / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )

    total_hours = (time.time() - t_global) / 3600
    print(f"\n  All done! Total wall time: {total_hours:.1f}h", flush=True)
    print(f"  Results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
