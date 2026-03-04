#!/usr/bin/env python3
"""Analyze the high-resolution phase transition data from the 7M hires experiment.

Produces the key plots for the descriptive paper:
  1. Per-behavior ρ vs training step (phase transition curves)
  2. Grassmann angle trajectory vs step (geometric reorganization)
  3. Effective dimensionality trajectory vs step
  4. Transition sharpness analysis (sigmoid fits + transition widths)

Usage:
    python experiments/scale_ladder/analyze_phase_transition.py
    python experiments/scale_ladder/analyze_phase_transition.py --results-dir results/scale_ladder/7M_seed42_hires
    python experiments/scale_ladder/analyze_phase_transition.py --quick  # Just ρ trajectories, no geometry
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_trajectory(results_dir):
    """Load all checkpoint audit results from hires experiment.

    Uses phase_transition_trajectory.json for audit data if available,
    then enriches with subspace data from individual checkpoint directories.
    Always scans checkpoint dirs for subspace_report.json files that may
    have been generated separately (e.g. by extract_transition_geometry.py).
    """
    results_dir = Path(results_dir)
    trajectory = []

    # Try combined trajectory file first for audit data
    traj_path = results_dir / "phase_transition_trajectory.json"
    if traj_path.exists():
        trajectory = json.loads(traj_path.read_text())
        trajectory = sorted(trajectory, key=lambda x: x["step"])
    else:
        # Fall back to scanning checkpoint dirs for audit data
        for ckpt_dir in sorted(results_dir.glob("checkpoint_*")):
            step = int(ckpt_dir.name.split("_")[-1])
            audit_path = ckpt_dir / "audit_report.json"
            if not audit_path.exists():
                continue
            entry = {
                "step": step,
                "audit": json.loads(audit_path.read_text()),
            }
            trajectory.append(entry)
        trajectory = sorted(trajectory, key=lambda x: x["step"])

    # Enrich with subspace data from individual checkpoint dirs
    # (may have been generated separately from audit data)
    step_to_idx = {e["step"]: i for i, e in enumerate(trajectory)}
    n_enriched = 0
    for ckpt_dir in sorted(results_dir.glob("checkpoint_*")):
        step = int(ckpt_dir.name.split("_")[-1])
        subspace_path = ckpt_dir / "subspace_report.json"
        if not subspace_path.exists():
            continue

        if step in step_to_idx:
            idx = step_to_idx[step]
            if "subspace" not in trajectory[idx]:
                trajectory[idx]["subspace"] = json.loads(subspace_path.read_text())
                n_enriched += 1
        else:
            # Subspace exists but no audit — include anyway for geometry analysis
            entry = {
                "step": step,
                "subspace": json.loads(subspace_path.read_text()),
            }
            trajectory.append(entry)
            n_enriched += 1

    if n_enriched > 0:
        trajectory = sorted(trajectory, key=lambda x: x["step"])

    return trajectory


def extract_rho_trajectories(trajectory):
    """Extract per-behavior ρ trajectories from audit results.

    Returns: dict of {behavior: [(step, rho), ...]}
    """
    rho_traj = {}

    for entry in trajectory:
        step = entry["step"]
        results = {r["behavior"]: r["rho"] for r in entry["audit"]["results"]}

        for beh, rho in results.items():
            if beh not in rho_traj:
                rho_traj[beh] = []
            rho_traj[beh].append((step, rho))

    # Sort each trajectory by step
    for beh in rho_traj:
        rho_traj[beh] = sorted(rho_traj[beh], key=lambda x: x[0])

    return rho_traj


def extract_geometry_trajectories(trajectory):
    """Extract Grassmann angle and effective dim trajectories.

    Returns:
        angles: dict of {behavior_pair: [(step, angle), ...]} for final-layer angles
        eff_dims: dict of {behavior: [(step, final_layer_eff_dim), ...]}
        sv_ratios: dict of {behavior: [(step, sv1/sv2_ratio), ...]}
    """
    import ast
    import re

    angles = {}
    eff_dims = {}
    sv_ratios = {}

    for entry in trajectory:
        if "subspace" not in entry:
            continue

        step = entry["step"]
        subspace = entry["subspace"]
        n_layers = subspace.get("n_layers", 0)
        final_layer = str(n_layers - 1)

        # Extract final-layer Grassmann angles
        overlap = subspace.get("overlap", {})
        if final_layer in overlap:
            data = overlap[final_layer]
            if isinstance(data, str) and "subspace_angles=" in data:
                m = re.search(r"subspace_angles=(\[\[.*?\]\])", data)
                if m:
                    matrix = ast.literal_eval(m.group(1))
                    # Get behavior names from the overlap string
                    beh_m = re.search(r"behaviors=\[(.*?)\]", data)
                    if beh_m:
                        beh_names = [b.strip().strip("'\"") for b in beh_m.group(1).split(",")]
                    else:
                        beh_names = subspace.get("behaviors", [])

                    for i in range(len(matrix)):
                        for j in range(i + 1, len(matrix)):
                            pair = f"{beh_names[i]}↔{beh_names[j]}" if len(beh_names) > j else f"pair_{i}_{j}"
                            if pair not in angles:
                                angles[pair] = []
                            angles[pair].append((step, matrix[i][j]))

        # Extract final-layer effective dimensionality
        eff_dim_data = subspace.get("effective_dimensionality", {})
        for beh_name, layer_dict in eff_dim_data.items():
            if final_layer in layer_dict:
                layer_data = layer_dict[final_layer]
                dim_val = layer_data.get("effective_dim", 0)
                if beh_name not in eff_dims:
                    eff_dims[beh_name] = []
                eff_dims[beh_name].append((step, dim_val))

                # SV1/SV2 ratio
                svs = layer_data.get("singular_values_top5", [])
                if len(svs) >= 2 and svs[1] > 1e-8:
                    ratio = svs[0] / svs[1]
                else:
                    ratio = float("inf")
                if beh_name not in sv_ratios:
                    sv_ratios[beh_name] = []
                sv_ratios[beh_name].append((step, ratio))

    # Sort all trajectories
    for d in [angles, eff_dims, sv_ratios]:
        for key in d:
            d[key] = sorted(d[key], key=lambda x: x[0])

    return angles, eff_dims, sv_ratios


def fit_sigmoid(steps, values, behavior_name=""):
    """Fit a sigmoid to a behavioral ρ trajectory.

    Returns dict with:
        midpoint: step at which ρ crosses 50% of [min, max]
        width: transition width (steps for 10%→90% of range)
        amplitude: max - min
        fit_quality: R² of the fit
        params: (L, k, x0, b) for L / (1 + exp(-k*(x-x0))) + b
    """
    from scipy.optimize import curve_fit

    steps = np.array(steps, dtype=float)
    values = np.array(values, dtype=float)

    # Skip if no transition (all same value or too few points)
    if len(steps) < 4 or np.std(values) < 0.01:
        return None

    # Sigmoid function
    def sigmoid(x, L, k, x0, b):
        return L / (1.0 + np.exp(-k * (x - x0))) + b

    # Initial guesses
    L0 = np.max(values) - np.min(values)
    b0 = np.min(values)
    x0_guess = steps[len(steps) // 2]
    k0 = 0.01  # Gentle slope initial guess

    try:
        popt, _ = curve_fit(
            sigmoid, steps, values,
            p0=[L0, k0, x0_guess, b0],
            bounds=(
                [0, 0, steps[0], -1],  # Lower bounds
                [2, 1, steps[-1], 2],  # Upper bounds
            ),
            maxfev=5000,
        )
        L, k, x0, b = popt

        # Compute fit quality (R²)
        predicted = sigmoid(steps, *popt)
        ss_res = np.sum((values - predicted) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-12)

        # Transition width: steps for 10%→90% of amplitude
        # sigmoid crosses 10% at x0 - ln(9)/k, 90% at x0 + ln(9)/k
        if k > 1e-6:
            width = 2 * np.log(9) / k
        else:
            width = float("inf")

        return {
            "behavior": behavior_name,
            "midpoint": float(x0),
            "width": float(width),
            "amplitude": float(L),
            "baseline": float(b),
            "steepness": float(k),
            "fit_quality_r2": float(r_squared),
            "params": [float(p) for p in popt],
        }
    except (RuntimeError, ValueError) as e:
        return {
            "behavior": behavior_name,
            "error": str(e),
            "fit_quality_r2": 0.0,
        }


def analyze_transitions(rho_traj):
    """Fit sigmoids to all behavioral ρ trajectories.

    Returns dict with per-behavior transition analysis.
    """
    results = {}

    for beh, points in rho_traj.items():
        steps = [p[0] for p in points]
        values = [p[1] for p in points]

        # Simple threshold crossing detection
        first_nonzero = None
        for s, v in points:
            if v > 0.05:
                first_nonzero = s
                break

        half_max = None
        max_val = max(values)
        for s, v in points:
            if v > max_val * 0.5:
                half_max = s
                break

        entry = {
            "behavior": beh,
            "min_rho": float(min(values)),
            "max_rho": float(max(values)),
            "range": float(max(values) - min(values)),
            "first_nonzero_step": first_nonzero,
            "half_max_step": half_max,
            "n_points": len(points),
        }

        # Sigmoid fit (needs scipy)
        try:
            fit = fit_sigmoid(steps, values, beh)
            if fit and "error" not in fit:
                entry["sigmoid_fit"] = fit
        except ImportError:
            pass

        results[beh] = entry

    return results


def generate_plots(rho_traj, angles, eff_dims, sv_ratios, transitions, output_dir):
    """Generate all phase transition plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Color scheme for behaviors
    behavior_colors = {
        "overrefusal": "#e74c3c",    # Red
        "refusal": "#c0392b",        # Dark red
        "factual": "#3498db",        # Blue
        "toxicity": "#2ecc71",       # Green
        "bias": "#9b59b6",           # Purple
        "sycophancy": "#f39c12",     # Orange
        "deception": "#e67e22",      # Dark orange
        "reasoning": "#1abc9c",      # Teal
    }

    # ─────────────────────────────────────────────────────────────────
    # Plot 1: Per-behavior ρ vs step (THE main phase transition plot)
    # ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))

    for beh, points in sorted(rho_traj.items()):
        steps = [p[0] for p in points]
        values = [p[1] for p in points]
        color = behavior_colors.get(beh, "#95a5a6")
        ax.plot(steps, values, "o-", label=beh, color=color, linewidth=2, markersize=4)

        # Add sigmoid fit if available
        if beh in transitions and "sigmoid_fit" in transitions[beh]:
            fit = transitions[beh]["sigmoid_fit"]
            if fit.get("fit_quality_r2", 0) > 0.8:
                L, k, x0, b = fit["params"]
                x_smooth = np.linspace(min(steps), max(steps), 200)
                y_smooth = L / (1 + np.exp(-k * (x_smooth - x0))) + b
                ax.plot(x_smooth, y_smooth, "--", color=color, alpha=0.5, linewidth=1)
                # Mark midpoint
                ax.axvline(x=x0, color=color, alpha=0.2, linestyle=":")

    # Annotate the known transition zone
    ax.axvspan(1850, 2050, alpha=0.08, color="red", label="overrefusal transition (steps 1900–2000)")

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("ρ (Spearman)", fontsize=12)
    ax.set_title("Behavioral Phase Transitions — 7M Model (High-Resolution)", fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    fig.savefig(output_dir / "phase_transition_rho.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'phase_transition_rho.png'}")

    # ─────────────────────────────────────────────────────────────────
    # Plot 2: Overrefusal zoom (the sharpest transition)
    # ─────────────────────────────────────────────────────────────────
    if "overrefusal" in rho_traj:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: overrefusal ρ with sigmoid fit
        points = rho_traj["overrefusal"]
        steps = [p[0] for p in points]
        values = [p[1] for p in points]
        ax1.plot(steps, values, "o-", color="#e74c3c", linewidth=2, markersize=5)

        if "overrefusal" in transitions and "sigmoid_fit" in transitions["overrefusal"]:
            fit = transitions["overrefusal"]["sigmoid_fit"]
            if "params" in fit:
                L, k, x0, b = fit["params"]
                x_smooth = np.linspace(min(steps), max(steps), 200)
                y_smooth = L / (1 + np.exp(-k * (x_smooth - x0))) + b
                ax1.plot(x_smooth, y_smooth, "--", color="black", linewidth=1.5,
                         label=f"sigmoid (midpoint={x0:.0f}, width={fit['width']:.0f})")
                ax1.axvline(x=x0, color="gray", linestyle=":", alpha=0.5)
                ax1.legend(fontsize=9)

        ax1.set_xlabel("Training Step", fontsize=12)
        ax1.set_ylabel("ρ (Spearman)", fontsize=12)
        ax1.set_title("Overrefusal Phase Transition", fontsize=13)
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)

        # Right: rate of change (numerical derivative)
        if len(steps) > 2:
            drho = np.gradient(values, steps)
            ax2.plot(steps, drho, "s-", color="#e74c3c", linewidth=1.5, markersize=4)
            ax2.set_xlabel("Training Step", fontsize=12)
            ax2.set_ylabel("dρ/dstep", fontsize=12)
            ax2.set_title("Transition Rate (Overrefusal)", fontsize=13)
            ax2.grid(True, alpha=0.3)

            # Mark the peak
            peak_idx = np.argmax(np.abs(drho))
            ax2.annotate(
                f"peak at step {steps[peak_idx]}",
                xy=(steps[peak_idx], drho[peak_idx]),
                xytext=(steps[peak_idx] + 200, drho[peak_idx] * 0.8),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=10,
            )

        plt.tight_layout()
        fig.savefig(output_dir / "phase_transition_overrefusal_zoom.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_dir / 'phase_transition_overrefusal_zoom.png'}")

    # ─────────────────────────────────────────────────────────────────
    # Plot 3: Grassmann angle trajectory (if geometry data available)
    # ─────────────────────────────────────────────────────────────────
    if angles:
        fig, ax = plt.subplots(figsize=(12, 7))

        for pair, points in sorted(angles.items()):
            steps = [p[0] for p in points]
            values = [p[1] for p in points]
            ax.plot(steps, values, "o-", label=pair, linewidth=1.5, markersize=3)

        ax.axhline(y=90.0, color="gray", linestyle=":", alpha=0.3, label="orthogonal")
        ax.axvspan(1850, 2050, alpha=0.08, color="red")
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Grassmann Angle (degrees)", fontsize=12)
        ax.set_title("Subspace Separation During Phase Transition", fontsize=14)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / "phase_transition_angles.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_dir / 'phase_transition_angles.png'}")

    # ─────────────────────────────────────────────────────────────────
    # Plot 4: Effective dimensionality trajectory
    # ─────────────────────────────────────────────────────────────────
    if eff_dims:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for beh, points in sorted(eff_dims.items()):
            steps = [p[0] for p in points]
            values = [p[1] for p in points]
            color = behavior_colors.get(beh, "#95a5a6")
            ax1.plot(steps, values, "o-", label=beh, color=color, linewidth=1.5, markersize=3)

        ax1.axvspan(1220, 2440, alpha=0.05, color="red")
        ax1.set_xlabel("Training Step", fontsize=12)
        ax1.set_ylabel("Effective Dimensionality", fontsize=12)
        ax1.set_title("Subspace Rank During Phase Transition", fontsize=13)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # SV1/SV2 ratio (concentration)
        for beh, points in sorted(sv_ratios.items()):
            steps = [p[0] for p in points]
            values = [min(p[1], 50) for p in points]  # Clip inf
            color = behavior_colors.get(beh, "#95a5a6")
            ax2.plot(steps, values, "o-", label=beh, color=color, linewidth=1.5, markersize=3)

        ax2.axvspan(1220, 2440, alpha=0.05, color="red")
        ax2.set_xlabel("Training Step", fontsize=12)
        ax2.set_ylabel("SV₁/SV₂ Ratio", fontsize=12)
        ax2.set_title("Singular Value Concentration", fontsize=13)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_dir / "phase_transition_geometry.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_dir / 'phase_transition_geometry.png'}")

    # ─────────────────────────────────────────────────────────────────
    # Plot 5: Combined transition + geometry (key paper figure)
    # ─────────────────────────────────────────────────────────────────
    if angles or eff_dims:
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        # Panel A: ρ trajectories
        for beh, points in sorted(rho_traj.items()):
            steps = [p[0] for p in points]
            values = [p[1] for p in points]
            color = behavior_colors.get(beh, "#95a5a6")
            axes[0].plot(steps, values, "o-", label=beh, color=color, linewidth=1.5, markersize=3)
        axes[0].set_ylabel("ρ (Spearman)", fontsize=11)
        axes[0].set_title("A. Behavioral Signal", fontsize=12)
        axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-0.05, 1.05)

        # Panel B: Grassmann angles
        if angles:
            for pair, points in sorted(angles.items()):
                steps = [p[0] for p in points]
                values = [p[1] for p in points]
                axes[1].plot(steps, values, "o-", label=pair, linewidth=1.2, markersize=2)
            axes[1].axhline(y=90.0, color="gray", linestyle=":", alpha=0.3)
            axes[1].set_ylabel("Grassmann Angle (°)", fontsize=11)
            axes[1].set_title("B. Subspace Separation (Final Layer)", fontsize=12)
            axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
            axes[1].grid(True, alpha=0.3)

        # Panel C: Effective dimensionality
        if eff_dims:
            for beh, points in sorted(eff_dims.items()):
                steps = [p[0] for p in points]
                values = [p[1] for p in points]
                color = behavior_colors.get(beh, "#95a5a6")
                axes[2].plot(steps, values, "o-", label=beh, color=color, linewidth=1.5, markersize=3)
            axes[2].set_ylabel("Effective Dim", fontsize=11)
            axes[2].set_title("C. Subspace Rank (Final Layer)", fontsize=12)
            axes[2].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
            axes[2].grid(True, alpha=0.3)

        axes[-1].set_xlabel("Training Step", fontsize=11)

        # Add transition zone shading to all panels
        for ax in axes:
            ax.axvspan(1850, 2050, alpha=0.08, color="red")

        plt.tight_layout()
        fig.savefig(output_dir / "phase_transition_combined.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_dir / 'phase_transition_combined.png'}")

    # ─────────────────────────────────────────────────────────────────
    # Plot 6: Factual decline analysis
    # ─────────────────────────────────────────────────────────────────
    if "factual" in rho_traj:
        fig, ax = plt.subplots(figsize=(10, 5))

        points = rho_traj["factual"]
        steps = [p[0] for p in points]
        values = [p[1] for p in points]
        ax.plot(steps, values, "o-", color="#3498db", linewidth=2, markersize=5, label="factual ρ")

        # Annotate peak
        peak_idx = np.argmax(values)
        ax.annotate(
            f"peak: ρ={values[peak_idx]:.3f} at step {steps[peak_idx]}",
            xy=(steps[peak_idx], values[peak_idx]),
            xytext=(steps[peak_idx] + 300, values[peak_idx] - 0.05),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=10,
        )

        # Add overrefusal for context
        if "overrefusal" in rho_traj:
            oref_points = rho_traj["overrefusal"]
            ax.plot(
                [p[0] for p in oref_points],
                [p[1] for p in oref_points],
                "o-", color="#e74c3c", linewidth=1.5, markersize=3, alpha=0.5,
                label="overrefusal ρ (context)",
            )

        ax.axvspan(1850, 2050, alpha=0.08, color="red")
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("ρ (Spearman)", fontsize=12)
        ax.set_title("Factual Knowledge Trajectory — Evidence for Geometric Priors", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / "phase_transition_factual_decline.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_dir / 'phase_transition_factual_decline.png'}")


def print_summary(rho_traj, transitions, angles=None, eff_dims=None):
    """Print a text summary of the phase transition analysis."""

    print(f"\n{'═' * 70}")
    print(f"  PHASE TRANSITION ANALYSIS SUMMARY")
    print(f"{'═' * 70}")

    # Behavioral ρ summary
    print(f"\n  Per-Behavior ρ Trajectory:")
    print(f"  {'Behavior':<15s} {'Start':>8s} {'End':>8s} {'Peak':>8s} {'Peak Step':>10s} {'Range':>8s}")
    print(f"  {'─' * 65}")
    for beh, points in sorted(rho_traj.items()):
        values = [p[1] for p in points]
        steps = [p[0] for p in points]
        peak_idx = np.argmax(values)
        print(
            f"  {beh:<15s} {values[0]:>8.3f} {values[-1]:>8.3f} "
            f"{values[peak_idx]:>8.3f} {steps[peak_idx]:>10d} "
            f"{max(values) - min(values):>8.3f}"
        )

    # Transition analysis
    transitioning = {k: v for k, v in transitions.items() if v["range"] > 0.1}
    if transitioning:
        print(f"\n  Behaviors with Phase Transitions (range > 0.1):")
        print(f"  {'Behavior':<15s} {'First>0.05':>10s} {'50% Step':>10s} {'Width':>8s} {'R²':>6s}")
        print(f"  {'─' * 55}")
        for beh, t in sorted(transitioning.items(), key=lambda x: x[1].get("first_nonzero_step", 99999)):
            width = ""
            r2 = ""
            if "sigmoid_fit" in t:
                fit = t["sigmoid_fit"]
                if "width" in fit:
                    width = f"{fit['width']:.0f}"
                if "fit_quality_r2" in fit:
                    r2 = f"{fit['fit_quality_r2']:.3f}"
            first = t.get("first_nonzero_step", "—")
            half = t.get("half_max_step", "—")
            print(f"  {beh:<15s} {str(first):>10s} {str(half):>10s} {width:>8s} {r2:>6s}")

    # Static behaviors
    static = {k: v for k, v in transitions.items() if v["range"] <= 0.1}
    if static:
        print(f"\n  Static Behaviors (range ≤ 0.1):")
        for beh, t in sorted(static.items()):
            print(f"    {beh}: ρ = {t['min_rho']:.3f}–{t['max_rho']:.3f}")

    # Geometry summary
    if angles:
        print(f"\n  Grassmann Angle Trajectories (final layer):")
        for pair, points in sorted(angles.items()):
            values = [p[1] for p in points]
            print(f"    {pair}: {min(values):.1f}° → {max(values):.1f}° "
                  f"(Δ={max(values)-min(values):.1f}°)")

    if eff_dims:
        print(f"\n  Effective Dim Trajectories (final layer):")
        for beh, points in sorted(eff_dims.items()):
            values = [p[1] for p in points]
            print(f"    {beh}: {min(values)} → {max(values)}")

    print(f"\n{'═' * 70}")


def main():
    parser = argparse.ArgumentParser(
        prog="analyze-phase-transition",
        description="Analyze high-resolution phase transition data from the 7M hires experiment",
    )
    parser.add_argument(
        "--results-dir", type=str,
        default="results/scale_ladder/7M_seed42_hires",
        help="Directory containing hires checkpoint audits",
    )
    parser.add_argument(
        "--figures-dir", type=str,
        default="figures/scale_ladder/phase_transition",
        help="Directory for output plots",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Only analyze ρ trajectories (skip geometry)",
    )
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  Phase Transition Analysis")
    print(f"  Results: {args.results_dir}")
    print(f"{'=' * 60}\n")

    # Load data
    trajectory = load_trajectory(args.results_dir)
    if not trajectory:
        print("  ERROR: No audited checkpoints found. Run audit first:")
        print(f"    python experiments/scale_ladder/phase_transition_hires.py --audit-only")
        return 1

    n_with_geometry = sum(1 for e in trajectory if "subspace" in e)
    print(f"  Loaded {len(trajectory)} checkpoints "
          f"({n_with_geometry} with geometry data)")
    print(f"  Step range: {trajectory[0]['step']}–{trajectory[-1]['step']}")

    # Extract trajectories
    rho_traj = extract_rho_trajectories(trajectory)

    angles, eff_dims, sv_ratios = {}, {}, {}
    if not args.quick:
        angles, eff_dims, sv_ratios = extract_geometry_trajectories(trajectory)

    # Analyze transitions
    transitions = analyze_transitions(rho_traj)

    # Print summary
    print_summary(rho_traj, transitions, angles, eff_dims)

    # Generate plots
    print(f"\n  --- Generating Plots ---")
    generate_plots(rho_traj, angles, eff_dims, sv_ratios, transitions, args.figures_dir)

    # Save analysis results
    output = {
        "n_checkpoints": len(trajectory),
        "step_range": [trajectory[0]["step"], trajectory[-1]["step"]],
        "rho_trajectories": {
            beh: [{"step": s, "rho": v} for s, v in points]
            for beh, points in rho_traj.items()
        },
        "transitions": transitions,
    }
    if angles:
        output["angle_trajectories"] = {
            pair: [{"step": s, "angle": v} for s, v in points]
            for pair, points in angles.items()
        }
    if eff_dims:
        output["eff_dim_trajectories"] = {
            beh: [{"step": s, "eff_dim": v} for s, v in points]
            for beh, points in eff_dims.items()
        }

    output_path = Path(args.results_dir) / "phase_transition_analysis.json"
    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n  Saved: {output_path}")

    print(f"\n{'=' * 60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
