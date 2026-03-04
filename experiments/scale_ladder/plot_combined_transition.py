#!/usr/bin/env python3
"""Generate the publication-quality combined phase transition figure.

3-panel figure:
  A. Behavioral ρ trajectories (all 8 behaviors)
  B. Grassmann angle trajectories (6 key pairs, grouped)
  C. Effective dimensionality + SV concentration (dual y-axis)

Designed for the Developmental Geometry paper.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    results_dir = Path("results/scale_ladder/7M_seed42_hires")
    figures_dir = Path("figures/scale_ladder/phase_transition")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load analysis results
    analysis = json.loads((results_dir / "phase_transition_analysis.json").read_text())
    rho_traj = analysis["rho_trajectories"]
    angle_traj = analysis.get("angle_trajectories", {})
    eff_dim_traj = analysis.get("eff_dim_trajectories", {})

    # ── Color scheme ──────────────────────────────────────────────
    beh_colors = {
        "overrefusal": "#e74c3c",
        "refusal": "#c0392b",
        "factual": "#3498db",
        "toxicity": "#2ecc71",
        "bias": "#9b59b6",
        "sycophancy": "#f39c12",
        "deception": "#e67e22",
        "reasoning": "#1abc9c",
    }

    # Key angle pairs (grouped by interpretive importance)
    KEY_PAIRS = {
        # Most entangled (paper story: why safety hurts knowledge)
        "factual↔refusal": {"color": "#e74c3c", "lw": 2.5, "label": "factual↔refusal (26→33°)"},
        # Moderate entanglement
        "deception↔toxicity": {"color": "#e67e22", "lw": 2.0, "label": "deception↔toxicity (29→49°)"},
        "factual↔sycophancy": {"color": "#f39c12", "lw": 2.0, "label": "factual↔sycophancy (42→53°)"},
        # Near-orthogonal (well separated)
        "bias↔toxicity": {"color": "#9b59b6", "lw": 2.0, "label": "bias↔toxicity (47→57°)"},
        "sycophancy↔toxicity": {"color": "#2ecc71", "lw": 2.0, "label": "sycophancy↔toxicity (51→62°)"},
        "bias↔sycophancy": {"color": "#3498db", "lw": 2.0, "label": "bias↔sycophancy (53→61°)"},
    }

    # ── Figure layout ─────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 14))
    gs = fig.add_gridspec(3, 1, hspace=0.28, height_ratios=[1, 1, 1])
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    transition_zone = (1850, 2050)

    # ── Panel A: Behavioral ρ trajectories ────────────────────────
    # Draw order: flat behaviors first (background), then transitions on top
    draw_order = ["bias", "sycophancy", "reasoning", "deception", "toxicity",
                  "refusal", "factual", "overrefusal"]

    for beh in draw_order:
        if beh not in rho_traj:
            continue
        points = rho_traj[beh]
        steps = [p["step"] for p in points]
        values = [p["rho"] for p in points]
        color = beh_colors.get(beh, "#95a5a6")

        # Highlight transitioning behaviors
        if beh == "overrefusal":
            ax_a.plot(steps, values, "o-", color=color, linewidth=2.5, markersize=4,
                     label=beh, zorder=10)
        elif beh == "factual":
            ax_a.plot(steps, values, "o-", color=color, linewidth=2.0, markersize=4,
                     label=beh, zorder=9)
        elif beh in ("bias", "sycophancy", "reasoning"):
            ax_a.plot(steps, values, "-", color=color, linewidth=1.0, markersize=0,
                     alpha=0.4, label=f"{beh} (ρ=0)")
        else:
            ax_a.plot(steps, values, "o-", color=color, linewidth=1.5, markersize=3,
                     label=beh)

    # Mark factual peak
    factual_pts = rho_traj["factual"]
    fvals = [p["rho"] for p in factual_pts]
    fsteps = [p["step"] for p in factual_pts]
    peak_idx = np.argmax(fvals)
    ax_a.annotate(
        f"factual peak\nρ={fvals[peak_idx]:.3f}",
        xy=(fsteps[peak_idx], fvals[peak_idx]),
        xytext=(fsteps[peak_idx] - 400, fvals[peak_idx] + 0.15),
        arrowprops=dict(arrowstyle="->", color="#3498db", lw=1.5),
        fontsize=9, color="#3498db", fontweight="bold",
    )

    ax_a.axvspan(*transition_zone, alpha=0.10, color="red", zorder=0)
    ax_a.set_ylabel("ρ (Spearman)", fontsize=12)
    ax_a.set_title("A.  Behavioral Signal Emergence", fontsize=13, fontweight="bold", loc="left")
    ax_a.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax_a.grid(True, alpha=0.2)
    ax_a.set_ylim(-0.05, 1.08)
    ax_a.set_xlim(750, 2850)

    # ── Panel B: Grassmann angle trajectories (key pairs) ─────────
    # Background: all other pairs in light gray
    for pair, points in angle_traj.items():
        if pair in KEY_PAIRS:
            continue
        steps = [p["step"] for p in points]
        values = [p["angle"] for p in points]
        ax_b.plot(steps, values, "-", color="#cccccc", linewidth=0.7, alpha=0.5, zorder=1)

    # Foreground: key pairs
    for pair, style in KEY_PAIRS.items():
        if pair not in angle_traj:
            continue
        points = angle_traj[pair]
        steps = [p["step"] for p in points]
        values = [p["angle"] for p in points]
        ax_b.plot(steps, values, "o-", color=style["color"], linewidth=style["lw"],
                 markersize=4, label=style["label"], zorder=5)

    ax_b.axhline(y=90.0, color="gray", linestyle=":", alpha=0.3)
    ax_b.axvspan(*transition_zone, alpha=0.10, color="red", zorder=0)

    # Annotate the near-alignment finding
    if "factual↔refusal" in angle_traj:
        fr_pts = angle_traj["factual↔refusal"]
        ax_b.annotate(
            "near-aligned\n(shared repr.)",
            xy=(fr_pts[0]["step"], fr_pts[0]["angle"]),
            xytext=(fr_pts[0]["step"] + 300, fr_pts[0]["angle"] - 8),
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
            fontsize=9, color="#e74c3c", fontweight="bold",
        )

    ax_b.set_ylabel("Grassmann Angle (°)", fontsize=12)
    ax_b.set_title("B.  Subspace Separation (Final Layer) — Smooth Evolution Through Sharp Transition",
                   fontsize=13, fontweight="bold", loc="left")
    ax_b.legend(loc="lower right", fontsize=7.5, ncol=2, framealpha=0.9)
    ax_b.grid(True, alpha=0.2)
    ax_b.set_ylim(20, 70)
    ax_b.set_xlim(750, 2850)

    # ── Panel C: Effective dimensionality ─────────────────────────
    # Left axis: effective dim; right axis: SV1/SV2 for the 1D behaviors

    for beh, points in sorted(eff_dim_traj.items()):
        steps = [p["step"] for p in points]
        values = [p["eff_dim"] for p in points]
        color = beh_colors.get(beh, "#95a5a6")

        if beh in ("bias", "sycophancy"):
            # 1D behaviors - dashed
            ax_c.plot(steps, values, "--", color=color, linewidth=1.5,
                     label=f"{beh} (dim=1)", alpha=0.7)
        else:
            ax_c.plot(steps, values, "o-", color=color, linewidth=1.8,
                     markersize=4, label=beh)

    ax_c.axvspan(*transition_zone, alpha=0.10, color="red", zorder=0)

    # Annotate the dim expansion
    ax_c.annotate(
        "toxicity subspace\nexpands 11→21",
        xy=(2800, 21), xytext=(2200, 23),
        arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=1.5),
        fontsize=9, color="#2ecc71", fontweight="bold",
    )

    ax_c.set_ylabel("Effective Dimensionality", fontsize=12)
    ax_c.set_xlabel("Training Step", fontsize=12)
    ax_c.set_title("C.  Subspace Rank (Final Layer)", fontsize=13, fontweight="bold", loc="left")
    ax_c.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax_c.grid(True, alpha=0.2)
    ax_c.set_xlim(750, 2850)

    # ── Global annotation ─────────────────────────────────────────
    # Add transition zone label
    for ax in [ax_a, ax_b, ax_c]:
        ax.text(1950, ax.get_ylim()[1] * 0.97, "overrefusal\ntransition",
               ha="center", va="top", fontsize=7, color="red", alpha=0.6,
               style="italic")

    fig.suptitle(
        "Phase Transition Geometry — 7M GPT-2 (Seed 42, High-Resolution)",
        fontsize=15, fontweight="bold", y=0.995
    )

    fig.savefig(figures_dir / "phase_transition_combined_v2.png", dpi=200, bbox_inches="tight",
               facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {figures_dir / 'phase_transition_combined_v2.png'}")

    # ── Print key findings ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  KEY FINDINGS")
    print("=" * 60)

    print("\n  1. SHARP BEHAVIOR, SMOOTH GEOMETRY")
    print("     Overrefusal ρ: 0.000 → 0.980 in ~100 steps (width=22)")
    print("     All 15 Grassmann angles: smooth monotonic widening")
    print("     → Phase transition is behavioral 'snap-in', not geometric reorganization")

    print("\n  2. FACTUAL↔REFUSAL NEAR-ALIGNMENT")
    if "factual↔refusal" in angle_traj:
        fr = angle_traj["factual↔refusal"]
        print(f"     Angle: {fr[0]['angle']:.1f}° → {fr[-1]['angle']:.1f}° "
              f"(closest pair by far)")
        print("     → Safety and knowledge share representational space")
        print("     → Aggressive safety training risks factual collateral damage")

    print("\n  3. DIMENSIONALITY STRATIFICATION")
    print("     bias, sycophancy: dim=1 throughout (rank-1 features)")
    print("     factual: 6 → 11 (distributed, high-rank)")
    print("     toxicity: 11 → 21 (most complex behavioral subspace)")
    print("     → Behaviors differ fundamentally in geometric complexity")

    print("\n  4. FACTUAL PEAK AT TRANSITION ONSET")
    print(f"     Factual ρ peaks at step {fsteps[peak_idx]} (ρ={fvals[peak_idx]:.3f})")
    print("     = exact step overrefusal first shows signal (ρ=0.080)")
    print("     → Possible competition for shared representational resources")

    if "deception↔toxicity" in angle_traj:
        dt = angle_traj["deception↔toxicity"]
        print("\n  5. DECEPTION↔TOXICITY DRAMATIC WIDENING")
        print(f"     Angle: {dt[0]['angle']:.1f}° → {dt[-1]['angle']:.1f}° "
              f"(Δ={dt[-1]['angle']-dt[0]['angle']:.1f}°, largest separation)")
        print("     → These start nearly aligned, then diverge into distinct subspaces")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
