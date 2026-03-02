#!/usr/bin/env python3
"""CatSAE re-run with Age weight override to 1.0x minimum.

Reuses saved census from the first CatSAE run, overrides Age vulnerability
so its final weight is >= 1.0x after normalization, then runs surgery.
"""
import sys, json, time
from pathlib import Path

# Ensure experiments/ and src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cat_sae_mlx import (
    build_weighted_protection,
    run_catsae_surgery,
    mlx_svd_compress,
)

MODEL = "Qwen/Qwen2.5-7B-Instruct"
CENSUS_PATH = Path("results/overnight_sweep/Qwen2.5-7B-Instruct/cat_sae/category_census.json")
OUTPUT_DIR = Path("results/overnight_sweep/Qwen2.5-7B-Instruct/cat_sae_age_override")
GAMMA = 0.10
RHO = 0.2
COMPRESS_RATIO = 0.7

CATEGORIES = [
    "Age", "Gender_biology", "Race_ethnicity",
    "Sexual_orientation_biology", "Religion",
]

# Age weight floor: we want final normalized weight >= 1.0x
# The formula is: raw_weight = 0.5 + 2.0 * vuln_score
# Then normalized by dividing by mean.
# To guarantee Age >= 1.0x post-normalization, we set its vuln_score
# high enough. With 5 categories, the exact score depends on others.
# Simplest: set Age vuln_score = 0.5 (gives raw weight 1.5), which
# after normalization with the other scores will be ~1.0x.
# But we'll also clamp post-hoc to be safe.
AGE_VULN_OVERRIDE = 0.5  # was 0.162


def main():
    t0 = time.time()

    print(f"\n{'='*70}", flush=True)
    print(f"  CatSAE Age Override — Weight Floor 1.0x", flush=True)
    print(f"  Model:      {MODEL}", flush=True)
    print(f"  Categories: {CATEGORIES}", flush=True)
    print(f"  Age vuln:   0.162 → {AGE_VULN_OVERRIDE} (override)", flush=True)
    print(f"  γ={GAMMA}, ρ={RHO}, compress={COMPRESS_RATIO}", flush=True)
    print(f"{'='*70}\n", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load saved census ──────────────────────────────────────
    print("  Loading saved census...", flush=True)
    census = json.loads(CENSUS_PATH.read_text())
    vulnerability = census["vulnerability"]

    # Override Age
    old_score = vulnerability["Age"]["vulnerability_score"]
    vulnerability["Age"]["vulnerability_score"] = AGE_VULN_OVERRIDE
    print(f"  Age vulnerability: {old_score:.4f} → {AGE_VULN_OVERRIDE}", flush=True)

    # ── Build weighted protection ──────────────────────────────
    print(f"\n  Building protection dataset with override...", flush=True)
    protection_dataset, weights, cat_counts = build_weighted_protection(
        vulnerability, CATEGORIES,
        base_pairs_per_category=50,
    )

    # Post-hoc clamp: if Age is still < 1.0x, renormalize
    if weights.get("Age", 0) < 1.0:
        print(f"  Age weight {weights['Age']:.3f}x < 1.0x, clamping...", flush=True)
        weights["Age"] = 1.0
        # Renormalize others so mean stays ~1.0
        others = [c for c in weights if c != "Age"]
        other_sum = sum(weights[c] for c in others)
        target_other_sum = len(weights) - 1.0  # total should be len(weights)
        if other_sum > 0:
            scale = target_other_sum / other_sum
            for c in others:
                weights[c] *= scale
        print(f"  Renormalized weights:", flush=True)
        for cat in sorted(weights, key=weights.get, reverse=True):
            print(f"    {cat:<35s} {weights[cat]:.3f}x", flush=True)

    print(f"\n  Total protection pairs: {len(protection_dataset)}", flush=True)
    for cat, n in sorted(cat_counts.items()):
        print(f"    {cat:<35s} {n} pairs", flush=True)

    # ── Load model ─────────────────────────────────────────────
    print(f"\n  Loading model...", flush=True)
    import mlx_lm
    model, tokenizer = mlx_lm.load(MODEL)
    print(f"  Model loaded.", flush=True)

    # ── Surgery ────────────────────────────────────────────────
    print(f"\n  Running surgery (γ={GAMMA}, ρ={RHO})...", flush=True)
    surgery_result = run_catsae_surgery(
        model, tokenizer, MODEL,
        protection_dataset=protection_dataset,
        gamma_weight=GAMMA,
        rho_weight=RHO,
        compress_ratio=COMPRESS_RATIO,
    )

    # ── Verification ───────────────────────────────────────────
    print(f"\n  Verification:", flush=True)
    baseline_rho = surgery_result["baseline"]["rho"]
    final_rho = surgery_result["final"]["rho"]
    print(f"  Overall: {baseline_rho:.4f} → {final_rho:.4f} "
          f"(Δ={final_rho - baseline_rho:+.4f})", flush=True)

    baseline_cats = surgery_result["baseline"].get("category_metrics", {})
    final_cats = surgery_result["final"].get("category_metrics", {})

    print(f"\n  {'Category':<30s} {'Before':>7s} {'After':>7s} "
          f"{'Delta':>7s} {'Weight':>6s}", flush=True)
    print(f"  {'-'*58}", flush=True)

    for cat in sorted(final_cats.keys()):
        before = baseline_cats.get(cat, {}).get("accuracy", 0)
        after = final_cats[cat]["accuracy"]
        delta = after - before
        w = weights.get(cat, 0.0)
        marker = "FAIL" if delta < -0.05 else ("+" if delta > 0.05 else " ")
        print(f"  {cat:<30s} {before:>6.1%} {after:>6.1%} "
              f"{delta:>+6.1%} {w:>5.2f}x {marker}", flush=True)

    # ── Save ───────────────────────────────────────────────────
    elapsed = time.time() - t0
    result = {
        "model": MODEL,
        "experiment": "catsae_age_override",
        "config": {
            "gamma": GAMMA,
            "rho": RHO,
            "compress_ratio": COMPRESS_RATIO,
            "age_vuln_override": AGE_VULN_OVERRIDE,
            "categories": CATEGORIES,
        },
        "protection_weights": {c: round(w, 3) for c, w in weights.items()},
        "protection_pair_counts": cat_counts,
        "surgery": {
            "baseline_bias": surgery_result["baseline"],
            "final_bias": surgery_result["final"],
            "sft_result": surgery_result["sft_result"],
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    out_path = OUTPUT_DIR / "surgery_result.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n  Saved: {out_path}", flush=True)
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
