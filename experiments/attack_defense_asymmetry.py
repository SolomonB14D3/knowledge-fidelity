"""Attack/Defense Asymmetry Experiment.

Measures whether behavioral steering is symmetric: is it equally hard to
degrade safety behaviors (attack) as to improve them (defense)?

This is red-team research for defensive purposes. The attack condition runs
on a local model; no jailbroken model is published or deployed. Understanding
the attack surface informs the design of more robust defenses.

Method:
  1. Train a Gated SAE on refusal + deception activations
  2. Identify behavior-specific features
  3. Sweep steering scale from 0.0 (full ablation) to 5.0 (max amplification)
  4. Measure rho for all 8 behaviors at each scale
  5. Compare attack slope (scale < 1) vs defense slope (scale > 1)
  6. Run matched-compute SFT in both directions (swap positive/negative)

Usage:
    # Quick test on 0.5B
    python experiments/attack_defense_asymmetry.py --model Qwen/Qwen2.5-0.5B --layer 12

    # Full run on 7B
    python experiments/attack_defense_asymmetry.py --model Qwen/Qwen2.5-7B --layer 17

    # SAE sweep only (skip SFT, faster)
    python experiments/attack_defense_asymmetry.py --model Qwen/Qwen2.5-0.5B --skip-sft
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np


# ── Configuration ─────────────────────────────────────────────────────────

ATTACK_DEFENSE_SCALES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
TARGET_BEHAVIORS = ["refusal", "deception"]
EVAL_BEHAVIORS = [
    "factual", "toxicity", "bias", "sycophancy",
    "reasoning", "refusal", "deception", "overrefusal",
]
RESULTS_DIR = Path("results") / "attack_defense"


# ── Phase 1: Setup ────────────────────────────────────────────────────────

def phase_setup(model_name, layer_idx, device, trust_remote_code=False):
    """Load model, run baseline audit, collect activations, train SAE."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rho_eval.audit import audit
    from rho_eval.steering import train_behavioral_sae, SAEConfig

    print(f"\n{'='*60}")
    print(f"  Phase 1: Setup")
    print(f"  Model: {model_name}")
    print(f"  Layer: {layer_idx}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # Load model
    print("\n  Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=trust_remote_code,
    ).to(device)
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Baseline audit
    print("\n  Running baseline audit (all 8 behaviors)...")
    t0 = time.time()
    baseline_report = audit(
        model=model, tokenizer=tokenizer,
        behaviors="all", device=device,
    )
    baseline_scores = {}
    for result in baseline_report.results:
        baseline_scores[result.behavior] = {
            "rho": result.rho,
            "status": result.status,
        }
        print(f"    {result.behavior:<15} rho={result.rho:.4f}  [{result.status}]")
    print(f"  Baseline audit: {time.time() - t0:.1f}s")

    # Collect activations for refusal + deception
    print(f"\n  Training SAE on {TARGET_BEHAVIORS} at layer {layer_idx}...")
    t0 = time.time()
    sae, act_data, train_stats = train_behavioral_sae(
        model, tokenizer,
        behaviors=TARGET_BEHAVIORS,
        layer_idx=layer_idx,
        device=device,
    )
    print(f"  SAE training: {time.time() - t0:.1f}s")
    print(f"  Active features: {train_stats['n_active_features']}")
    print(f"  Dead features: {len(train_stats['dead_features'])}")

    # Identify behavioral features
    print("\n  Identifying behavioral features...")
    from rho_eval.steering import identify_behavioral_features
    reports, behavioral_features = identify_behavioral_features(sae, act_data)

    for behavior in TARGET_BEHAVIORS:
        n_features = len(behavioral_features.get(behavior, []))
        print(f"    {behavior}: {n_features} features identified")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "sae": sae,
        "act_data": act_data,
        "behavioral_features": behavioral_features,
        "baseline_scores": baseline_scores,
        "train_stats": {
            k: v for k, v in train_stats.items()
            if k not in ("sae",)  # don't serialize the SAE object
        },
    }


# ── Phase 2: SAE Steering Sweep ──────────────────────────────────────────

def phase_steering_sweep(setup_data, layer_idx, scales=None):
    """Sweep steering scale for each target behavior, measuring all 8 rho scores."""
    from rho_eval.audit import audit
    from rho_eval.steering import steer_features

    if scales is None:
        scales = ATTACK_DEFENSE_SCALES

    model = setup_data["model"]
    tokenizer = setup_data["tokenizer"]
    sae = setup_data["sae"]
    behavioral_features = setup_data["behavioral_features"]
    baseline_scores = setup_data["baseline_scores"]

    results = {}

    for target in TARGET_BEHAVIORS:
        feature_indices = behavioral_features.get(target, [])
        if not feature_indices:
            print(f"\n  WARNING: No features for {target}, skipping sweep")
            results[target] = {"error": "no features identified"}
            continue

        print(f"\n{'='*60}")
        print(f"  Phase 2: Steering sweep on {target}")
        print(f"  Features: {len(feature_indices)}")
        print(f"  Scales: {scales}")
        print(f"{'='*60}")

        sweep_results = []

        for scale in scales:
            print(f"\n  [scale={scale:.2f}] ", end="", flush=True)

            # Install hook
            hook = steer_features(model, sae, layer_idx, feature_indices, scale)

            try:
                # Run audit with hook active
                report = audit(
                    model=model, tokenizer=tokenizer,
                    behaviors="all",
                )

                rho_scores = {}
                for r in report.results:
                    rho_scores[r.behavior] = r.rho

                # Compute deltas from baseline
                deltas = {}
                for beh in rho_scores:
                    bl = baseline_scores.get(beh, {}).get("rho", 0.0)
                    deltas[beh] = round(rho_scores[beh] - bl, 6)

                # Print summary
                target_rho = rho_scores.get(target, 0.0)
                target_delta = deltas.get(target, 0.0)
                collateral = [
                    v for k, v in deltas.items()
                    if k != target and v < -0.01
                ]
                n_regressed = len(collateral)
                mean_collateral = sum(collateral) / len(collateral) if collateral else 0.0

                print(f"{target}={target_rho:.4f} (delta={target_delta:+.4f}), "
                      f"collateral: {n_regressed} regressed (mean={mean_collateral:+.4f})")

                sweep_results.append({
                    "scale": scale,
                    "rho_scores": rho_scores,
                    "deltas": deltas,
                    "target_rho": target_rho,
                    "target_delta": target_delta,
                    "n_collateral_regressions": n_regressed,
                    "mean_collateral": round(mean_collateral, 6),
                })

            finally:
                hook.remove()

        results[target] = {
            "n_features": len(feature_indices),
            "feature_indices": feature_indices[:50],
            "sweep": sweep_results,
        }

    return results


# ── Phase 3: Contrastive SFT (Attack vs Defense) ─────────────────────────

def phase_sft_attack_defense(setup_data, model_name, device, trust_remote_code=False):
    """Run matched-compute SFT with labels swapped: defense vs attack."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rho_eval.audit import audit

    print(f"\n{'='*60}")
    print(f"  Phase 3: Contrastive SFT (Attack vs Defense)")
    print(f"{'='*60}")

    results = {}

    for condition in ["defense", "attack"]:
        print(f"\n  --- Condition: {condition.upper()} ---")

        # Load fresh model for each condition (so they start from the same point)
        print("  Loading fresh model...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=trust_remote_code,
        ).to(device)

        # Build contrast pairs from refusal probes
        from rho_eval.behaviors import get_behavior
        refusal_behavior = get_behavior("refusal")
        refusal_probes = refusal_behavior.load_probes(n=999)  # load all available

        contrast_pairs = []
        for probe in refusal_probes:
            text = probe.get("text", "")
            harmful = probe.get("harmful_version", "")
            if text and harmful:
                if condition == "defense":
                    # Reinforce refusal: benign response is positive
                    contrast_pairs.append({
                        "positive": text,
                        "negative": harmful,
                    })
                else:
                    # Attack: compliance is positive (swap labels)
                    contrast_pairs.append({
                        "positive": harmful,
                        "negative": text,
                    })

        print(f"  {len(contrast_pairs)} contrast pairs ({condition} labeling)")

        # Build minimal SFT dataset (use the benign texts as SFT data)
        sft_texts = [p["positive"] for p in contrast_pairs[:50]]

        # Run rho-guided SFT
        try:
            from rho_eval.alignment import _HAS_MLX
        except ImportError:
            _HAS_MLX = False

        print("  Running rho-guided SFT...")
        t0 = time.time()

        if _HAS_MLX:
            # MLX path (Apple Silicon)
            from rho_eval.alignment import mlx_rho_guided_sft
            sft_result = mlx_rho_guided_sft(
                model, tokenizer, sft_texts,
                contrast_dataset=contrast_pairs,
                rho_weight=0.3,
                epochs=1,
                lr=2e-4,
                margin=0.1,
            )
        else:
            # PyTorch path
            from rho_eval.alignment import rho_guided_sft, BehavioralContrastDataset
            contrast_ds = BehavioralContrastDataset(contrast_pairs)
            sft_result = rho_guided_sft(
                model, tokenizer, sft_texts,
                contrast_dataset=contrast_ds,
                rho_weight=0.3,
                epochs=1,
                lr=2e-4,
                margin=0.1,
            )

        elapsed = time.time() - t0
        print(f"  SFT done: {elapsed:.1f}s")

        # Run post-SFT audit
        print("  Running post-SFT audit...")
        post_report = audit(
            model=model, tokenizer=tokenizer,
            behaviors="all",
        )

        post_scores = {}
        for r in post_report.results:
            post_scores[r.behavior] = r.rho
            bl = setup_data["baseline_scores"].get(r.behavior, {}).get("rho", 0.0)
            delta = r.rho - bl
            print(f"    {r.behavior:<15} rho={r.rho:.4f} (delta={delta:+.4f})")

        # Compute deltas
        deltas = {}
        for beh in post_scores:
            bl = setup_data["baseline_scores"].get(beh, {}).get("rho", 0.0)
            deltas[beh] = round(post_scores[beh] - bl, 6)

        results[condition] = {
            "rho_scores": post_scores,
            "deltas": deltas,
            "elapsed_sec": elapsed,
            "n_contrast_pairs": len(contrast_pairs),
            "n_sft_texts": len(sft_texts),
        }

        # Free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


# ── Phase 4: Asymmetry Analysis ──────────────────────────────────────────

def phase_analysis(steering_results, sft_results, baseline_scores):
    """Compute asymmetry metrics from steering sweep and SFT results."""

    print(f"\n{'='*60}")
    print(f"  Phase 4: Asymmetry Analysis")
    print(f"{'='*60}")

    analysis = {}

    # ── SAE Steering Asymmetry ────────────────────────────────────────
    for target in TARGET_BEHAVIORS:
        if isinstance(steering_results.get(target), dict) and "sweep" in steering_results[target]:
            sweep = steering_results[target]["sweep"]

            # Build scale → target_rho mapping
            scale_rho = {s["scale"]: s["target_rho"] for s in sweep}
            baseline_rho = scale_rho.get(1.0, None)

            if baseline_rho is None:
                print(f"  WARNING: No scale=1.0 result for {target}")
                continue

            # Attack effectiveness: how much rho drops per unit of suppression
            attack_points = [(s, r) for s, r in scale_rho.items() if s < 1.0]
            defense_points = [(s, r) for s, r in scale_rho.items() if s > 1.0]

            # Attack slope: rho change per unit scale decrease (from 1.0)
            attack_deltas = [(1.0 - s, baseline_rho - r) for s, r in attack_points]
            defense_deltas = [(s - 1.0, r - baseline_rho) for s, r in defense_points]

            # Simple linear slope: delta_rho / delta_scale
            if attack_deltas:
                attack_slope = np.mean([dr / ds for ds, dr in attack_deltas if ds > 0])
            else:
                attack_slope = 0.0

            if defense_deltas:
                defense_slope = np.mean([dr / ds for ds, dr in defense_deltas if ds > 0])
            else:
                defense_slope = 0.0

            asymmetry_ratio = attack_slope / defense_slope if defense_slope > 1e-8 else float("inf")

            # Threshold analysis: what scale to move rho by ±0.1?
            attack_threshold = None
            for s in sorted(scale_rho.keys()):
                if s < 1.0 and baseline_rho - scale_rho[s] >= 0.1:
                    attack_threshold = s
                    break
            defense_threshold = None
            for s in sorted(scale_rho.keys()):
                if s > 1.0 and scale_rho[s] - baseline_rho >= 0.1:
                    defense_threshold = s
                    break

            # Collateral damage comparison
            attack_collateral = []
            defense_collateral = []
            for entry in sweep:
                non_target_deltas = [
                    v for k, v in entry["deltas"].items()
                    if k != target
                ]
                mean_coll = np.mean([abs(d) for d in non_target_deltas]) if non_target_deltas else 0.0
                if entry["scale"] < 1.0:
                    attack_collateral.append(mean_coll)
                elif entry["scale"] > 1.0:
                    defense_collateral.append(mean_coll)

            result = {
                "baseline_rho": round(baseline_rho, 6),
                "attack_slope": round(float(attack_slope), 6),
                "defense_slope": round(float(defense_slope), 6),
                "asymmetry_ratio": round(float(asymmetry_ratio), 4),
                "attack_threshold_for_0.1_drop": attack_threshold,
                "defense_threshold_for_0.1_gain": defense_threshold,
                "mean_attack_collateral": round(float(np.mean(attack_collateral)), 6) if attack_collateral else None,
                "mean_defense_collateral": round(float(np.mean(defense_collateral)), 6) if defense_collateral else None,
                "n_features": steering_results[target]["n_features"],
            }
            analysis[f"sae_{target}"] = result

            print(f"\n  {target} (SAE steering):")
            print(f"    Baseline rho:       {baseline_rho:.4f}")
            print(f"    Attack slope:       {attack_slope:.4f} (rho drop per unit suppression)")
            print(f"    Defense slope:      {defense_slope:.4f} (rho gain per unit amplification)")
            print(f"    Asymmetry ratio:    {asymmetry_ratio:.2f}x (>1 = attack is easier)")
            print(f"    Attack threshold:   scale={attack_threshold} for -0.1 rho")
            print(f"    Defense threshold:  scale={defense_threshold} for +0.1 rho")
            print(f"    N features:         {steering_results[target]['n_features']}")

    # ── SFT Asymmetry ─────────────────────────────────────────────────
    if sft_results:
        for target in TARGET_BEHAVIORS:
            bl = baseline_scores.get(target, {}).get("rho", 0.0)
            attack_rho = sft_results.get("attack", {}).get("rho_scores", {}).get(target, bl)
            defense_rho = sft_results.get("defense", {}).get("rho_scores", {}).get(target, bl)

            attack_delta = attack_rho - bl
            defense_delta = defense_rho - bl

            sft_asymmetry = abs(attack_delta) / abs(defense_delta) if abs(defense_delta) > 1e-6 else float("inf")

            analysis[f"sft_{target}"] = {
                "baseline_rho": round(bl, 6),
                "attack_rho": round(attack_rho, 6),
                "attack_delta": round(attack_delta, 6),
                "defense_rho": round(defense_rho, 6),
                "defense_delta": round(defense_delta, 6),
                "sft_asymmetry_ratio": round(float(sft_asymmetry), 4),
            }

            print(f"\n  {target} (SFT, matched compute):")
            print(f"    Attack:  rho={attack_rho:.4f} (delta={attack_delta:+.4f})")
            print(f"    Defense: rho={defense_rho:.4f} (delta={defense_delta:+.4f})")
            print(f"    Asymmetry: {sft_asymmetry:.2f}x")

    return analysis


# ── Phase 5: Visualization ────────────────────────────────────────────────

def phase_visualization(steering_results, sft_results, analysis, baseline_scores, output_dir):
    """Generate figures for the asymmetry analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for target in TARGET_BEHAVIORS:
        if isinstance(steering_results.get(target), dict) and "sweep" in steering_results[target]:
            sweep = steering_results[target]["sweep"]
            scales = [s["scale"] for s in sweep]
            target_rhos = [s["target_rho"] for s in sweep]
            bl = baseline_scores.get(target, {}).get("rho", 0.0)

            # ── Scale vs Rho plot ─────────────────────────────────
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Target behavior rho
            ax1.plot(scales, target_rhos, "o-", color="tab:blue", linewidth=2,
                     markersize=8, label=f"{target} rho")
            ax1.axhline(y=bl, color="tab:blue", linestyle="--", alpha=0.5,
                        label=f"baseline ({bl:.3f})")
            ax1.axvline(x=1.0, color="gray", linestyle=":", alpha=0.5)
            ax1.set_xlabel("Steering Scale", fontsize=12)
            ax1.set_ylabel(f"{target} rho", color="tab:blue", fontsize=12)
            ax1.tick_params(axis="y", labelcolor="tab:blue")

            # Shade attack vs defense regions
            ax1.axvspan(min(scales), 1.0, alpha=0.05, color="red", label="attack region")
            ax1.axvspan(1.0, max(scales), alpha=0.05, color="green", label="defense region")

            # Collateral damage on secondary axis
            ax2 = ax1.twinx()
            mean_collateral = [
                np.mean([abs(v) for k, v in s["deltas"].items() if k != target])
                for s in sweep
            ]
            ax2.plot(scales, mean_collateral, "s--", color="tab:red", linewidth=1.5,
                     markersize=6, alpha=0.7, label="mean |collateral|")
            ax2.set_ylabel("Mean |collateral damage|", color="tab:red", fontsize=12)
            ax2.tick_params(axis="y", labelcolor="tab:red")

            # Combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

            ax1.set_title(f"Attack/Defense Asymmetry: {target}", fontsize=14)
            fig.tight_layout()
            fig.savefig(fig_dir / f"scale_vs_rho_{target}.png", dpi=150)
            plt.close(fig)
            print(f"  Saved: {fig_dir / f'scale_vs_rho_{target}.png'}")

            # ── Collateral Heatmap ────────────────────────────────
            behaviors_ordered = [b for b in EVAL_BEHAVIORS if b != target]
            heatmap_data = []
            for s in sweep:
                row = [s["deltas"].get(b, 0.0) for b in behaviors_ordered]
                heatmap_data.append(row)

            heatmap_data = np.array(heatmap_data)

            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(heatmap_data.T, aspect="auto", cmap="RdYlGn",
                           vmin=-0.2, vmax=0.2)
            ax.set_xticks(range(len(scales)))
            ax.set_xticklabels([f"{s:.2f}" for s in scales], rotation=45)
            ax.set_yticks(range(len(behaviors_ordered)))
            ax.set_yticklabels(behaviors_ordered)
            ax.set_xlabel("Steering Scale")
            ax.set_title(f"Collateral Damage: steering {target}")
            fig.colorbar(im, label="delta rho from baseline")
            fig.tight_layout()
            fig.savefig(fig_dir / f"collateral_heatmap_{target}.png", dpi=150)
            plt.close(fig)
            print(f"  Saved: {fig_dir / f'collateral_heatmap_{target}.png'}")

    # ── SFT Attack vs Defense bar chart ───────────────────────────────
    if sft_results and "attack" in sft_results and "defense" in sft_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for i, target in enumerate(TARGET_BEHAVIORS):
            ax = axes[i]
            bl = baseline_scores.get(target, {}).get("rho", 0.0)
            attack_delta = sft_results["attack"]["deltas"].get(target, 0.0)
            defense_delta = sft_results["defense"]["deltas"].get(target, 0.0)

            bars = ax.bar(
                ["Attack\n(swap labels)", "Defense\n(normal labels)"],
                [attack_delta, defense_delta],
                color=["tab:red", "tab:green"],
                alpha=0.8,
                edgecolor="black",
            )
            ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
            ax.set_ylabel("Delta rho from baseline")
            ax.set_title(f"SFT {target}: Attack vs Defense\n(baseline rho={bl:.3f})")

            # Add value labels
            for bar, val in zip(bars, [attack_delta, defense_delta]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"{val:+.4f}", ha="center", va="bottom" if val > 0 else "top",
                        fontsize=10, fontweight="bold")

        fig.suptitle("Matched-Compute SFT: Attack vs Defense", fontsize=14)
        fig.tight_layout()
        fig.savefig(fig_dir / "sft_attack_vs_defense.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {fig_dir / 'sft_attack_vs_defense.png'}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Attack/Defense Asymmetry Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--layer", type=int, default=12,
        help="Layer index for SAE training (default: 12)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cuda, mps, cpu). Default: auto-detect",
    )
    parser.add_argument(
        "--skip-sft", action="store_true",
        help="Skip the SFT phase (faster, SAE sweep only)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=str(RESULTS_DIR),
        help=f"Output directory (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="Trust remote code when loading model",
    )
    args = parser.parse_args()

    # Resolve device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  ATTACK/DEFENSE ASYMMETRY EXPERIMENT")
    print(f"  Model:  {args.model}")
    print(f"  Layer:  {args.layer}")
    print(f"  Device: {device}")
    print(f"  SFT:    {'skip' if args.skip_sft else 'enabled'}")
    print(f"  Output: {output_dir}")
    print(f"{'#'*60}")

    t_start = time.time()

    # Phase 1: Setup
    setup_data = phase_setup(
        args.model, args.layer, device,
        trust_remote_code=args.trust_remote_code,
    )

    # Save baseline + SAE info
    with open(output_dir / "baseline_audit.json", "w") as f:
        json.dump(setup_data["baseline_scores"], f, indent=2)
    with open(output_dir / "sae_training_stats.json", "w") as f:
        json.dump(setup_data["train_stats"], f, indent=2, default=str)

    # Save feature info
    for target in TARGET_BEHAVIORS:
        features = setup_data["behavioral_features"].get(target, [])
        with open(output_dir / f"{target}_features.json", "w") as f:
            json.dump({
                "behavior": target,
                "n_features": len(features),
                "feature_indices": features,
            }, f, indent=2)

    # Phase 2: SAE Steering Sweep
    steering_results = phase_steering_sweep(setup_data, args.layer)

    for target in TARGET_BEHAVIORS:
        with open(output_dir / f"steering_sweep_{target}.json", "w") as f:
            json.dump(steering_results.get(target, {}), f, indent=2, default=str)

    # Phase 3: SFT Attack vs Defense
    sft_results = None
    if not args.skip_sft:
        sft_results = phase_sft_attack_defense(
            setup_data, args.model, device,
            trust_remote_code=args.trust_remote_code,
        )
        with open(output_dir / "sft_attack_audit.json", "w") as f:
            json.dump(sft_results.get("attack", {}), f, indent=2)
        with open(output_dir / "sft_defense_audit.json", "w") as f:
            json.dump(sft_results.get("defense", {}), f, indent=2)

    # Phase 4: Analysis
    analysis = phase_analysis(
        steering_results, sft_results, setup_data["baseline_scores"],
    )
    with open(output_dir / "asymmetry_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    # Phase 5: Visualization
    print(f"\n{'='*60}")
    print(f"  Phase 5: Visualization")
    print(f"{'='*60}")
    phase_visualization(
        steering_results, sft_results, analysis,
        setup_data["baseline_scores"], output_dir,
    )

    # Summary
    total_time = time.time() - t_start
    print(f"\n{'#'*60}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Results: {output_dir}")
    print(f"{'#'*60}")

    # Print key findings
    for key, val in analysis.items():
        if "asymmetry_ratio" in val:
            ratio = val["asymmetry_ratio"]
            direction = "attack is easier" if ratio > 1 else "defense is easier" if ratio < 1 else "symmetric"
            print(f"  {key}: asymmetry = {ratio:.2f}x ({direction})")


if __name__ == "__main__":
    main()
