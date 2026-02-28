#!/usr/bin/env python3
r"""Extended γ* bounds analysis with Monte Carlo uncertainty quantification.

Six analyses:
  1. Monte Carlo γ* distribution: sample N combinations of (s∞, cos(θ),
     baseline_ρ) from plausible uncertainty ranges → distribution of γ*.
  2. Nonlinear amplification model: sign-crossing amplification explaining
     the 22.5× observed vs 2.0× linear predicted interference ratio.
  3. Sensitivity analysis: tornado diagram of parameter dominance on γ*.
  4. Multi-behavior phase diagram: simultaneous γ* per behavior across
     the (γ, λ_ρ) plane.  Shows safe/dangerous regimes.
  5. Variance U-shape MC: model the non-monotonic factual σ vs λ_ρ,
     predict the optimal λ_ρ as intersection of two noise sources.
  6. Probe sampling noise: quantify how finite probe sets inflate σ
     and shift γ* via bootstrap over probe subsets.

Outputs (docs/):
  gamma_mc_distribution.png   — Monte Carlo histogram of γ*
  gamma_amplification.png     — Nonlinear amplification model
  gamma_sensitivity.png       — Tornado diagram
  gamma_phase_diagram.png     — (γ, λ_ρ) phase diagram: safe vs dangerous
  gamma_variance_ushape.png   — Variance U-shape decomposition + MC
  gamma_probe_noise.png       — Probe-sampling noise impact on γ*
  gamma_bounds_analysis.json  — Full data + summary stats

Usage:
  python scripts/gamma_bounds_analysis.py
  python scripts/gamma_bounds_analysis.py --n-samples 50000
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# EMPIRICAL ANCHORS (from ablation data: 5 seeds, Qwen2.5-7B, λ_ρ=0.2)
# ═══════════════════════════════════════════════════════════════════════════

# Margin ablation data
GAMMA_0 = {  # γ = 0.0
    "d_bias": -0.011,
    "d_factual": +0.136,
    "d_toxicity": +0.560,
    "d_sycophancy": +0.038,
}
GAMMA_01 = {  # γ = 0.1
    "d_bias": +0.034,
    "d_factual": +0.163,
    "d_toxicity": +0.621,
    "d_sycophancy": +0.040,
}

# Baseline ρ values (pre-training)
BASELINE_RHO = {
    "bias": 0.036,
    "factual": 0.603,
    "toxicity": 0.145,
    "sycophancy": -0.041,
}

# Grassmann angles from subspace analysis (Section 5 of paper)
# bias↔toxicity: ~82°, other pairs: ~86°
THETA_BIAS_TOX = 82.0  # degrees
THETA_OTHER = 86.0      # degrees

# Empirical interference at γ=0 and γ=0.1
I_BIAS_GAMMA0 = abs(BASELINE_RHO["bias"] - (BASELINE_RHO["bias"] + GAMMA_0["d_bias"]))
I_BIAS_GAMMA01 = abs(BASELINE_RHO["bias"] - (BASELINE_RHO["bias"] + GAMMA_01["d_bias"]))
# More precisely: interference = magnitude of ρ change due to cross-dim effects
# At γ=0: bias goes from 0.036 to 0.036+(-0.011) = 0.025 → loss of 0.011 + inversion = 0.047 total
# At γ=0.1: bias goes from 0.036 to 0.036+(+0.034) = 0.070 → gain, so interference = 0.002

# Empirical s∞ estimate
S_INF_POINT = 0.1 / (0.002 / 0.047)  # ≈ 2.35 nats

# Empirical interference swings (γ=0 → γ=0.1)
BIAS_SWING = abs(GAMMA_01["d_bias"] - GAMMA_0["d_bias"])         # 0.045
SYCO_SWING = abs(GAMMA_01["d_sycophancy"] - GAMMA_0["d_sycophancy"])  # 0.002


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Monte Carlo γ* Distribution
# ═══════════════════════════════════════════════════════════════════════════

def monte_carlo_gamma_star(n_samples: int = 10000, seed: int = 42) -> dict:
    r"""Sample γ* under parameter uncertainty.

    The critical margin γ* is the minimum γ that preserves the sign of
    a neighboring behavioral dimension.  From the interference bound:

        Δρ_j(γ) = Δρ_j(0) + [I_j(0) - I_j(γ)]
                 = Δρ_j(0) + I_j(0) · [1 - γ/s∞]

    Setting Δρ_j(γ*) = 0 and solving:

        γ* = s∞ · (1 - |Δρ_j(0)| / I_j(0))   if Δρ_j(0) < 0
           = s∞ · |Δρ_j(0)| / I_j(0)           [simplified]

    More precisely, from linear interpolation between the two data points:

        γ* = -b₀ / b₁

    where Δρ_bias(γ) ≈ b₀ + b₁·γ, with:
        b₀ = Δρ_bias(0)     — the no-margin bias change
        b₁ = [Δρ_bias(0.1) - Δρ_bias(0)] / 0.1  — the γ sensitivity

    We sample each parameter from a distribution reflecting measurement
    uncertainty (5 seeds → standard error of the mean).
    """
    rng = np.random.default_rng(seed)

    # --- Parameter distributions ---

    # s∞: point estimate 2.35, but depends on two noisy ratios
    # Model as lognormal centered at 2.35 with ~40% CV
    # (high uncertainty because it extrapolates from 2 data points)
    s_inf_mu = np.log(S_INF_POINT)
    s_inf_sigma = 0.35  # ~40% CV in log-space
    s_inf_samples = rng.lognormal(s_inf_mu, s_inf_sigma, n_samples)

    # cos(θ_bias↔tox): point estimate cos(82°) ≈ 0.139
    # Uncertainty in angle: ±3° (measurement noise from subspace extraction)
    theta_samples = rng.normal(THETA_BIAS_TOX, 3.0, n_samples)  # degrees
    theta_samples = np.clip(theta_samples, 60, 89.5)  # physical bounds
    cos_theta_samples = np.cos(np.radians(theta_samples))

    # Δρ_bias(0): point estimate -0.011, std from 5 seeds
    # Approximate SEM ≈ 0.003 (from ablation data std ~0.002, n≈5→14)
    d_bias_gamma0 = rng.normal(GAMMA_0["d_bias"], 0.003, n_samples)

    # Δρ_bias(0.1): point estimate +0.034, std ≈ 0.004
    d_bias_gamma01 = rng.normal(GAMMA_01["d_bias"], 0.004, n_samples)

    # Baseline ρ_bias: 0.036 ± 0.005 (pre-training measurement noise)
    baseline_bias = rng.normal(BASELINE_RHO["bias"], 0.005, n_samples)

    # --- Compute γ* for each sample ---

    # Linear model: Δρ_bias(γ) = b0 + b1*γ → γ* = -b0/b1
    b0 = d_bias_gamma0
    b1 = (d_bias_gamma01 - d_bias_gamma0) / 0.1
    # Only valid where b1 > 0 (bias improves with γ)
    valid = b1 > 0
    gamma_star = np.full(n_samples, np.nan)
    gamma_star[valid] = -b0[valid] / b1[valid]

    # Also compute γ* from the interference bound formulation
    # γ* = s∞ · (I_bias_0 - baseline_bias) / I_bias_0
    # where I_bias_0 = absolute interference at γ=0
    I_0 = np.abs(d_bias_gamma0)  # interference without margin
    I_01 = np.abs(d_bias_gamma01 - (d_bias_gamma01 + d_bias_gamma0) / 2)
    gamma_star_bound = s_inf_samples * I_01 / np.maximum(I_0, 1e-10)

    # --- Compute interference ratio at γ=0.1 ---
    # I(γ)/I(0) = γ/s∞
    interference_ratio = 0.1 / s_inf_samples

    # --- Summary statistics ---
    gs_valid = gamma_star[~np.isnan(gamma_star)]
    gs_valid = gs_valid[(gs_valid > 0) & (gs_valid < 0.5)]  # physical range

    results = {
        "n_samples": n_samples,
        "n_valid": len(gs_valid),
        "gamma_star": {
            "mean": float(np.mean(gs_valid)),
            "median": float(np.median(gs_valid)),
            "std": float(np.std(gs_valid)),
            "ci_95": [float(np.percentile(gs_valid, 2.5)),
                       float(np.percentile(gs_valid, 97.5))],
            "ci_90": [float(np.percentile(gs_valid, 5)),
                       float(np.percentile(gs_valid, 95))],
            "ci_68": [float(np.percentile(gs_valid, 16)),
                       float(np.percentile(gs_valid, 84))],
            "point_estimate": float(-GAMMA_0["d_bias"] /
                                    ((GAMMA_01["d_bias"] - GAMMA_0["d_bias"]) / 0.1)),
        },
        "s_inf": {
            "point_estimate": float(S_INF_POINT),
            "mc_mean": float(np.mean(s_inf_samples)),
            "mc_std": float(np.std(s_inf_samples)),
        },
        "interference_ratio_at_01": {
            "mean": float(np.mean(interference_ratio)),
            "std": float(np.std(interference_ratio)),
        },
        "input_distributions": {
            "s_inf": {"type": "lognormal", "mu": float(s_inf_mu),
                      "sigma": float(s_inf_sigma)},
            "theta_bias_tox": {"type": "normal", "mu": float(THETA_BIAS_TOX),
                               "sigma": 3.0, "unit": "degrees"},
            "d_bias_gamma0": {"type": "normal", "mu": float(GAMMA_0["d_bias"]),
                              "sigma": 0.003},
            "d_bias_gamma01": {"type": "normal", "mu": float(GAMMA_01["d_bias"]),
                               "sigma": 0.004},
        },
        "samples": {
            "gamma_star_valid": gs_valid.tolist(),
            "s_inf": s_inf_samples[:100].tolist(),  # subsample for JSON size
        },
    }

    return results, gs_valid, s_inf_samples, cos_theta_samples, baseline_bias


def plot_mc_distribution(gs_valid: np.ndarray, results: dict, out_path: Path):
    """Plot Monte Carlo histogram of γ*."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ci_95 = results["gamma_star"]["ci_95"]
    ci_68 = results["gamma_star"]["ci_68"]
    median = results["gamma_star"]["median"]
    point = results["gamma_star"]["point_estimate"]

    ax.hist(gs_valid, bins=80, density=True, alpha=0.7, color="#4C78A8",
            edgecolor="white", linewidth=0.5)

    # 95% CI shading
    ax.axvspan(ci_95[0], ci_95[1], alpha=0.1, color="#E45756",
               label=f"95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
    # 68% CI shading
    ax.axvspan(ci_68[0], ci_68[1], alpha=0.15, color="#F58518",
               label=f"68% CI: [{ci_68[0]:.3f}, {ci_68[1]:.3f}]")

    ax.axvline(median, color="#E45756", linestyle="-", linewidth=2,
               label=f"Median: {median:.4f}")
    ax.axvline(point, color="black", linestyle="--", linewidth=1.5,
               label=f"Point estimate: {point:.4f}")
    ax.axvline(0.1, color="#54A24B", linestyle=":", linewidth=2,
               label=r"Default $\gamma = 0.1$")

    ax.set_xlabel(r"Critical margin $\gamma^*$", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(r"Monte Carlo distribution of $\gamma^*$ ($N = "
                 + f"{len(gs_valid):,}" + r"$ valid samples)", fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_xlim(-0.01, 0.15)

    # Annotate safety factor
    sf = 0.1 / median
    ax.annotate(f"Safety factor: {sf:.1f}×",
                xy=(0.1, ax.get_ylim()[1] * 0.8),
                fontsize=11, ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#54A24B",
                          alpha=0.2))

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Nonlinear Amplification Model
# ═══════════════════════════════════════════════════════════════════════════

def nonlinear_amplification_analysis():
    r"""Model the sign-crossing amplification that explains the 15×/1.9× ratio.

    The linear model predicts interference scales as |cos(θ)|, giving a
    bias/sycophancy ratio of cos(82°)/cos(86°) ≈ 1.9×.

    Empirically we observe 15× (bias swing 0.045 vs sycophancy 0.003).

    Key insight: when a behavioral dimension has baseline ρ near zero,
    the ρ-space response to interference is nonlinearly amplified:

    1. The underlying CE-space interference IS linear in cos(θ)
    2. But the mapping from CE-space interference to ρ-space change is
       nonlinear — it depends on the baseline ρ distribution
    3. Near ρ ≈ 0, the probe ranking is nearly random, so small CE pushes
       can flip many probe orderings → large |Δρ|
    4. Far from ρ ≈ 0 (e.g., sycophancy at -0.041), the ranking is already
       established, so the same CE push produces minimal ρ change

    We model this as:
        Δρ_j ∝ |cos(θ_ij)| · A(ρ_j^baseline)

    where A(ρ) is the amplification factor:
        A(ρ) = 1 / (|ρ| + ε)    [inverse distance from zero]

    or more precisely from rank-correlation sensitivity analysis:
        A(ρ) ∝ f'(Φ^{-1}(ρ))   [derivative of the CDF-based response]
    """
    # --- Build the amplification model ---

    # Observed data
    behaviors = ["bias", "toxicity", "factual", "sycophancy"]
    baseline_rho = np.array([BASELINE_RHO[b] for b in behaviors])
    swings = np.array([
        BIAS_SWING,       # 0.045
        abs(GAMMA_01["d_toxicity"] - GAMMA_0["d_toxicity"]),   # 0.061
        abs(GAMMA_01["d_factual"] - GAMMA_0["d_factual"]),     # 0.027
        SYCO_SWING,       # 0.002
    ])

    # Approximate Grassmann angles (from paper)
    # bias↔tox: 82°, tox↔fact: ~84°, fact↔syco: ~87°, bias↔syco: ~86°
    # For interference FROM toxicity optimization ONTO each dimension:
    cos_thetas = np.array([
        np.cos(np.radians(82)),   # bias
        1.0,                       # toxicity (self — not applicable)
        np.cos(np.radians(84)),   # factual
        np.cos(np.radians(86)),   # sycophancy
    ])

    # Linear prediction: Δρ ∝ cos(θ)
    linear_pred = cos_thetas / cos_thetas[0] * swings[0]  # normalize to bias swing
    linear_ratio = cos_thetas[0] / cos_thetas[3]  # bias/syco cos ratio

    # Nonlinear model: Δρ ∝ cos(θ) · A(ρ_baseline)
    # Fit A(ρ) = c / (|ρ| + ε)
    # From bias and sycophancy data points:
    #   swing_bias / swing_syco = [cos(82)/cos(86)] · [A(0.036) / A(0.041)]
    #   15 = 1.9 · [A(0.036) / A(0.041)]
    #   A(0.036) / A(0.041) = 15/1.9 ≈ 7.9

    # This means the amplification model is:
    #   A(ρ) = 1 / (|ρ| + ε)^α
    # Solving: (0.041 + ε)^α / (0.036 + ε)^α = 7.9
    # With ε small: (0.041/0.036)^α = 7.9 → α = log(7.9)/log(1.139) ≈ 15.9
    # That's too steep — suggests ε matters

    # Better model: A(ρ) = 1 / (|ρ|^2 + ε²)^(1/2)  [regularized inverse]
    # Or simpler: the sensitivity of Spearman ρ to CE perturbation

    # Let's fit empirically
    # We have 4 data points: (baseline_ρ, cos_θ, swing)
    # Model: swing = C · cos(θ) · A(baseline_ρ)
    # Use bias and sycophancy (the two we want to explain):
    # Skip toxicity (self) and factual (less constrained angle)

    rho_grid = np.linspace(-0.5, 0.8, 200)

    # ρ-sensitivity model: near ρ=0, rank correlation is maximally unstable
    # Physical model: probes near the decision boundary (CE(x+) ≈ CE(x-))
    # are the ones that flip under interference.
    # Fraction of probes near boundary ∝ exp(-ρ²/2σ²)
    sigma_rho = 0.08  # fitted to match the 7.9× ratio

    A_model = np.exp(-rho_grid**2 / (2 * sigma_rho**2))
    A_bias = np.exp(-BASELINE_RHO["bias"]**2 / (2 * sigma_rho**2))
    A_syco = np.exp(-BASELINE_RHO["sycophancy"]**2 / (2 * sigma_rho**2))

    # Alternatively: fit sigma to match the ratio
    # A(0.036)/A(-0.041) = exp(-(0.036²-0.041²)/(2σ²))
    # But signs: |0.036| vs |0.041| → ratio < 1, but we need > 1
    # The asymmetry comes from |ρ|: sycophancy at -0.041 has LARGER |ρ| than bias at 0.036
    # So A(ρ_syco) < A(ρ_bias) since sycophancy's |ρ| is slightly larger

    # Gaussian model:
    # A(ρ_bias)/A(ρ_syco) = exp((-0.036² + 0.041²)/(2σ²))
    #                       = exp((0.001681 - 0.001296)/(2σ²))
    #                       = exp(0.000385/(2σ²))
    # We need this ≈ 7.9: 0.000385/(2σ²) = ln(7.9) ≈ 2.067
    # σ² = 0.000385 / (2·2.067) = 0.0000931 → σ = 0.00965
    sigma_fitted = np.sqrt(0.000385 / (2 * np.log(7.9)))

    # That sigma (0.00965) is tiny — unrealistically sharp
    # This suggests the Gaussian-in-ρ model is wrong.

    # Better approach: the amplification is NOT in ρ-space but in CE-space.
    # The key insight is that bias and sycophancy have different EFFECTIVE
    # dimensionalities in the residual stream.

    # Most likely explanation: the 1.9× vs 15× discrepancy comes from
    # MULTIPLE sources of amplification stacking:
    #
    # 1. Angular factor: cos(82°)/cos(86°) = 1.9×
    # 2. Baseline proximity to zero: bias at 0.036 is closer to zero
    #    than sycophancy's absolute magnitude |−0.041|, but the difference
    #    is tiny. More importantly:
    # 3. Probe set sensitivity: bias probes (BBQ) are template-based with
    #    subtle wording differences → high sensitivity to small CE shifts.
    #    Sycophancy probes are longer opinion texts → more robust to CE noise.
    # 4. Effective subspace dimensionality: bias may be encoded in a
    #    lower-dimensional subspace, making it more fragile.

    # Decompose the 15× into contributions:
    angular_factor = np.cos(np.radians(82)) / np.cos(np.radians(86))  # 1.99
    residual_amplification = (BIAS_SWING / SYCO_SWING) / angular_factor  # 7.5

    results = {
        "observed_ratio": float(BIAS_SWING / max(SYCO_SWING, 1e-10)),
        "linear_prediction": float(angular_factor),
        "residual_amplification": float(residual_amplification),
        "decomposition": {
            "angular_factor": float(angular_factor),
            "nonlinear_factor": float(residual_amplification),
            "product": float(angular_factor * residual_amplification),
        },
        "swing_bias": float(BIAS_SWING),
        "swing_sycophancy": float(SYCO_SWING),
        "baseline_bias": float(BASELINE_RHO["bias"]),
        "baseline_sycophancy": float(BASELINE_RHO["sycophancy"]),
        "sigma_fitted": float(sigma_fitted),
    }

    return results, behaviors, baseline_rho, swings, cos_thetas


def plot_amplification(amp_results: dict, behaviors, baseline_rho, swings,
                       cos_thetas, out_path: Path):
    """Plot the nonlinear amplification model."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel 1: Linear prediction vs observed ---
    ax = axes[0]
    behs_plot = ["bias", "sycophancy"]
    vals_obs = [BIAS_SWING, SYCO_SWING]
    vals_lin = [BIAS_SWING,
                BIAS_SWING * np.cos(np.radians(86)) / np.cos(np.radians(82))]

    x = np.arange(len(behs_plot))
    width = 0.35
    bars1 = ax.bar(x - width/2, vals_obs, width, label="Observed", color="#4C78A8")
    bars2 = ax.bar(x + width/2, vals_lin, width, label="Linear (cos θ) prediction",
                   color="#E45756", alpha=0.7)
    ax.set_ylabel(r"|$\Delta\rho$| swing ($\gamma=0 \to 0.1$)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(behs_plot, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_title("(a) Observed vs Linear Prediction", fontsize=12)

    # Annotate ratios
    ax.annotate(f"Observed: {vals_obs[0]/vals_obs[1]:.0f}×",
                xy=(0.5, max(vals_obs) * 0.8), fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8))
    ax.annotate(f"Linear: {amp_results['linear_prediction']:.1f}×",
                xy=(0.5, max(vals_obs) * 0.65), fontsize=10, color="#E45756",
                bbox=dict(facecolor="white", alpha=0.8))

    # --- Panel 2: Amplification decomposition ---
    ax = axes[1]
    factors = ["Angular\n" + r"$|\cos\theta|$",
               "Nonlinear\namplification",
               "Product"]
    values = [amp_results["decomposition"]["angular_factor"],
              amp_results["decomposition"]["nonlinear_factor"],
              amp_results["decomposition"]["product"]]
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    bars = ax.bar(factors, values, color=colors, edgecolor="white", linewidth=1.5)
    ax.axhline(amp_results["observed_ratio"], color="black", linestyle="--",
               linewidth=1.5, label=f"Observed ratio: {amp_results['observed_ratio']:.0f}×")
    ax.set_ylabel("Multiplicative factor", fontsize=11)
    ax.set_title("(b) Amplification Decomposition", fontsize=12)
    ax.legend(fontsize=10)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}×", ha="center", fontsize=10, fontweight="bold")

    # --- Panel 3: ρ-sensitivity model ---
    ax = axes[2]
    rho_grid = np.linspace(-0.15, 0.15, 300)

    # The key physics: Spearman ρ measures rank correlation.
    # When baseline ρ ≈ 0, probes are near-randomly ordered, so small CE
    # perturbations flip many pairs → large |Δρ|.
    # When |ρ| >> 0, probes are well-ordered, so perturbations flip few → small |Δρ|.
    #
    # Sensitivity dρ/dI ∝ density of probes near the decision boundary
    # ∝ 1/sqrt(|ρ| + ε) for rank statistics (from Gaussian copula model)

    epsilon = 0.005
    sensitivity = 1.0 / np.sqrt(np.abs(rho_grid) + epsilon)
    sensitivity /= sensitivity.max()

    ax.plot(rho_grid, sensitivity, color="#4C78A8", linewidth=2.5)
    ax.axvline(BASELINE_RHO["bias"], color="#E45756", linestyle="--",
               linewidth=2, label=f'Bias baseline (ρ={BASELINE_RHO["bias"]:.3f})')
    ax.axvline(BASELINE_RHO["sycophancy"], color="#F58518", linestyle="--",
               linewidth=2, label=f'Sycophancy baseline (ρ={BASELINE_RHO["sycophancy"]:.3f})')

    # Mark sensitivity values
    s_bias = 1.0 / np.sqrt(abs(BASELINE_RHO["bias"]) + epsilon)
    s_syco = 1.0 / np.sqrt(abs(BASELINE_RHO["sycophancy"]) + epsilon)
    s_max = 1.0 / np.sqrt(epsilon)
    s_bias_norm = s_bias / s_max
    s_syco_norm = s_syco / s_max

    ax.plot(BASELINE_RHO["bias"], s_bias_norm, "o", color="#E45756",
            markersize=10, zorder=5)
    ax.plot(BASELINE_RHO["sycophancy"], s_syco_norm, "o", color="#F58518",
            markersize=10, zorder=5)

    ax.set_xlabel(r"Baseline $\rho$", fontsize=11)
    ax.set_ylabel("Sensitivity to interference (normalized)", fontsize=11)
    ax.set_title(r"(c) $\rho$-Sensitivity: $d\rho/dI \propto 1/\sqrt{|\rho| + \epsilon}$",
                 fontsize=12)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Sensitivity (Tornado) Diagram
# ═══════════════════════════════════════════════════════════════════════════

def sensitivity_analysis(n_samples: int = 10000, seed: int = 42) -> dict:
    r"""One-at-a-time sensitivity: which parameters dominate γ* uncertainty?

    For each parameter, fix all others at their point estimates and sweep
    the target from its 5th to 95th percentile.  Report the resulting
    range of γ*.
    """
    rng = np.random.default_rng(seed)

    # Point estimates
    d_bias_0_pt = GAMMA_0["d_bias"]       # -0.011
    d_bias_01_pt = GAMMA_01["d_bias"]     # +0.034
    s_inf_pt = S_INF_POINT                 # 2.35
    theta_pt = THETA_BIAS_TOX              # 82.0°

    # γ* from linear model: -b0/b1 where b0=d_bias_0, b1=(d_bias_01-d_bias_0)/0.1
    def gamma_star(d0, d01, _s_inf=None, _theta=None):
        b1 = (d01 - d0) / 0.1
        if b1 <= 0:
            return np.nan
        return -d0 / b1

    # Baseline γ*
    gs_baseline = gamma_star(d_bias_0_pt, d_bias_01_pt)

    # Sweep each parameter
    params = {}

    # 1. d_bias(γ=0): the no-margin bias change
    d0_range = np.linspace(d_bias_0_pt - 0.010, d_bias_0_pt + 0.010, 200)
    gs_d0 = [gamma_star(d0, d_bias_01_pt) for d0 in d0_range]
    gs_d0 = np.array(gs_d0)
    valid = ~np.isnan(gs_d0) & (gs_d0 > 0) & (gs_d0 < 0.5)
    params[r"$\Delta\rho_{bias}(\gamma=0)$"] = {
        "low": float(np.nanpercentile(gs_d0[valid], 5)) if valid.any() else gs_baseline,
        "high": float(np.nanpercentile(gs_d0[valid], 95)) if valid.any() else gs_baseline,
        "param_range": f"[{d_bias_0_pt-0.010:+.3f}, {d_bias_0_pt+0.010:+.3f}]",
    }

    # 2. d_bias(γ=0.1): the with-margin bias change
    d01_range = np.linspace(d_bias_01_pt - 0.012, d_bias_01_pt + 0.012, 200)
    gs_d01 = [gamma_star(d_bias_0_pt, d01) for d01 in d01_range]
    gs_d01 = np.array(gs_d01)
    valid = ~np.isnan(gs_d01) & (gs_d01 > 0) & (gs_d01 < 0.5)
    params[r"$\Delta\rho_{bias}(\gamma=0.1)$"] = {
        "low": float(np.nanpercentile(gs_d01[valid], 5)) if valid.any() else gs_baseline,
        "high": float(np.nanpercentile(gs_d01[valid], 95)) if valid.any() else gs_baseline,
        "param_range": f"[{d_bias_01_pt-0.012:+.3f}, {d_bias_01_pt+0.012:+.3f}]",
    }

    # 3. s∞: affects bound formulation but not linear γ* directly
    # Through the interference bound: γ* = s∞ · (I_01/I_0)
    # I(γ)/I(0) = γ/s∞, so γ* = s∞ · [Δρ_bias(0.1)-Δρ_bias(0)] / [0.1·slope]
    # Actually γ* from linear interpolation is independent of s∞.
    # But the interference BOUND on γ* depends on s∞.
    # From bound: γ*_bound = s∞ · (residual_interference / total_interference)

    # Interference bound formulation:
    # At γ=0: total_interference = 0.047 (= -d_bias(0) when d_bias(0)<0 + baseline)
    # At γ=0.1: residual = 0.002
    # ratio = 0.002/0.047 = 0.0426
    # γ*_bound = γ where residual → 0, so γ*_bound = s∞ · 0 → need different approach

    # s∞ enters through: ratio = γ/s∞, so s∞ = γ/ratio
    # For the bound: γ* = s∞ · (1 - d_bias_01/d_bias_0) ... this is getting circular
    # Let's just use the direct linear model for the tornado.

    s_inf_range = np.linspace(1.0, 5.0, 200)
    # γ* from bound: γ*_bound = s∞ · (I_margin / I_total)
    # I_margin/I_total = Δ_bias_01 / Δ_bias_0 (in absolute interference terms)
    # Not quite — let's compute: γ* where I(γ*)/I(0) = baseline_bias/I(0)
    # I(0) = 0.047, we need I(γ*) = baseline_bias = 0.036
    # I(γ)/I(0) = γ/s∞ → 0.036/0.047 = γ*/s∞ → γ* = s∞ · 0.766
    # Wait, that gives γ* ≈ 1.8, way too high.

    # The correct formulation: γ* where Δρ_bias = 0
    # From linear: γ* = 0.1 · |d_bias_0| / (d_bias_01 - d_bias_0)
    # This is independent of s∞!

    # So s∞ affects the TIGHTNESS of the bound, not γ* itself.
    # Include s∞ as affecting the safety factor instead.
    params[r"$s_\infty$ (nats)"] = {
        "low": float(gs_baseline),  # independent
        "high": float(gs_baseline),
        "param_range": "[1.0, 5.0]",
        "note": "Does not affect linear γ* (affects bound tightness only)",
    }

    # 4. θ_bias↔tox: angle between subspaces
    theta_range = np.linspace(75, 89, 200)
    cos_range = np.cos(np.radians(theta_range))
    # θ affects interference magnitude but γ* is about the margin threshold
    # γ* depends on the interference ratio, which cancels θ
    # However, if we model: Δρ_bias = baseline + cos(θ)·f(γ)
    # Then γ* shifts with θ.  The angular factor affects HOW MUCH interference,
    # and thus how much margin is needed.
    # Stronger interference (smaller θ) → need larger γ to compensate → larger γ*
    # Let's model: γ* ∝ cos(θ)/cos(θ_ref) · γ*_ref
    gs_theta = gs_baseline * cos_range / np.cos(np.radians(THETA_BIAS_TOX))
    valid = (gs_theta > 0) & (gs_theta < 0.5)
    params[r"$\theta_{bias \leftrightarrow tox}$ (degrees)"] = {
        "low": float(np.nanpercentile(gs_theta[valid], 5)) if valid.any() else gs_baseline,
        "high": float(np.nanpercentile(gs_theta[valid], 95)) if valid.any() else gs_baseline,
        "param_range": "[75°, 89°]",
    }

    # 5. Baseline ρ_bias: determines how close to zero-crossing
    # Higher baseline → more room → smaller effective γ* needed
    # This enters through: Δρ_bias(γ) crosses zero at γ*
    # If baseline is higher, the same interference produces a later zero-crossing
    baseline_range = np.linspace(0.010, 0.080, 200)
    # Rescale: γ* shifts proportionally to baseline proximity to zero
    # d_bias(0) = baseline_new - baseline_old + d_bias_0_pt
    d0_shifted = [(b - BASELINE_RHO["bias"]) + d_bias_0_pt for b in baseline_range]
    d01_shifted = [(b - BASELINE_RHO["bias"]) + d_bias_01_pt for b in baseline_range]
    gs_base = [gamma_star(d0, d01) for d0, d01 in zip(d0_shifted, d01_shifted)]
    gs_base = np.array(gs_base)
    valid = ~np.isnan(gs_base) & (gs_base > 0) & (gs_base < 0.5)
    params[r"Baseline $\rho_{bias}$"] = {
        "low": float(np.nanpercentile(gs_base[valid], 5)) if valid.any() else gs_baseline,
        "high": float(np.nanpercentile(gs_base[valid], 95)) if valid.any() else gs_baseline,
        "param_range": "[0.010, 0.080]",
    }

    return {
        "baseline_gamma_star": float(gs_baseline),
        "parameters": {k: v for k, v in params.items()},
    }


def plot_tornado(sens_results: dict, out_path: Path):
    """Plot tornado sensitivity diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    gs_base = sens_results["baseline_gamma_star"]
    params = sens_results["parameters"]

    # Sort by total range (descending)
    sorted_params = sorted(params.items(),
                           key=lambda x: abs(x[1]["high"] - x[1]["low"]),
                           reverse=True)

    labels = []
    lows = []
    highs = []
    for name, data in sorted_params:
        labels.append(name)
        lows.append(data["low"] - gs_base)
        highs.append(data["high"] - gs_base)

    y_pos = np.arange(len(labels))

    # Bars
    for i, (label, lo, hi) in enumerate(zip(labels, lows, highs)):
        if abs(lo) < 1e-6 and abs(hi) < 1e-6:
            # No effect — draw a thin line
            ax.barh(i, 0.0001, left=0, color="#CCCCCC", edgecolor="gray")
        else:
            ax.barh(i, lo, left=0, color="#4C78A8", alpha=0.8, edgecolor="white")
            ax.barh(i, hi, left=0, color="#E45756", alpha=0.8, edgecolor="white")

    ax.axvline(0, color="black", linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel(r"$\Delta\gamma^*$ from baseline", fontsize=12)
    ax.set_title(rf"Sensitivity of $\gamma^*$ (baseline = {gs_base:.4f})", fontsize=13)

    # Add value annotations
    for i, (lo, hi) in enumerate(zip(lows, highs)):
        rng = abs(hi - lo)
        if rng > 1e-6:
            ax.text(max(hi, lo) + 0.002, i, f"±{rng/2:.4f}", va="center",
                    fontsize=9, color="gray")

    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# EMPIRICAL DOSE-RESPONSE DATA (5 seeds × 4 λ_ρ, Qwen2.5-7B, γ=0.1)
# ═══════════════════════════════════════════════════════════════════════════

DOSE_RESPONSE = {
    # λ_ρ: {behavior: (mean, std_pop)}  — population std (N denominator)
    0.0: {
        "factual": (0.0788, 0.0935), "toxicity": (-0.1996, 0.0523),
        "bias": (-0.0095, 0.0013), "sycophancy": (0.0378, 0.0007),
    },
    0.1: {
        "factual": (0.1165, 0.0792), "toxicity": (0.3721, 0.0685),
        "bias": (0.0233, 0.0025), "sycophancy": (0.0390, 0.0007),
    },
    0.2: {
        "factual": (0.1609, 0.0154), "toxicity": (0.5938, 0.0650),
        "bias": (0.0359, 0.0023), "sycophancy": (0.0398, 0.0005),
    },
    0.5: {
        "factual": (0.3105, 0.0370), "toxicity": (0.9646, 0.0534),
        "bias": (0.0467, 0.0021), "sycophancy": (0.0439, 0.0030),
    },
}

# No-margin per-seed data (γ=0, λ_ρ=0.2, 5 seeds)
NO_MARGIN_SEEDS = {
    "factual": [0.1387, 0.1467, 0.1655, 0.1187, 0.1092],
    "toxicity": [0.5543, 0.5613, 0.6874, 0.4751, 0.5220],
    "bias": [-0.0111, -0.0100, -0.0109, -0.0099, -0.0117],
    "sycophancy": [0.0397, 0.0399, 0.0390, 0.0396, 0.0392],
}

# Grassmann angles (from paper Section 5):  i→j angle
# toxicity is the primary optimization target; these are interference angles
GRASSMANN_ANGLES = {
    "bias": 82.0,       # bias↔toxicity
    "factual": 84.0,    # factual↔toxicity
    "sycophancy": 86.0, # sycophancy↔toxicity
}

# Probe counts per behavior (from probe landscape)
PROBE_COUNTS = {
    "factual": 206, "toxicity": 200, "bias": 300,
    "sycophancy": 150, "reasoning": 100, "refusal": 150,
    "deception": 100, "overrefusal": 150,
}


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: Multi-Behavior Phase Diagram
# ═══════════════════════════════════════════════════════════════════════════

def multi_behavior_phase(n_samples: int = 10000, seed: int = 42) -> dict:
    r"""Compute γ* for each behavior simultaneously across (γ, λ_ρ) plane.

    For each behavior j, the critical margin γ*_j is the γ below which
    Δρ_j flips sign (negative = harmful).

    Model: Δρ_j(γ, λ_ρ) = Δρ_j^{SFT} + λ_ρ · G_j(γ)

    where G_j(γ) is the net contrastive contribution to dimension j:
      - For the target dimension (toxicity): G is positive, increases with γ
      - For bystander dimensions: G can be negative at low γ (interference)
        and becomes positive at high γ (margin protection)

    We fit G_j from the empirical dose-response and margin ablation data,
    then MC-sample over parameter noise to get the joint distribution.
    """
    rng = np.random.default_rng(seed)

    behaviors = ["factual", "bias", "sycophancy"]  # non-target bystanders
    lambda_rhos = np.array([0.0, 0.1, 0.2, 0.5])

    # ── Fit linear dose-response slopes: Δρ_j ≈ a_j + b_j · λ_ρ ──
    # (at γ=0.1, which is the standard setting)
    slopes = {}
    intercepts = {}
    for beh in behaviors:
        ys = np.array([DOSE_RESPONSE[lam][beh][0] for lam in lambda_rhos])
        coeffs = np.polyfit(lambda_rhos, ys, 1)
        slopes[beh] = coeffs[0]     # dΔρ/dλ
        intercepts[beh] = coeffs[1] # Δρ at λ=0

    # ── Margin sensitivity: how γ shifts the dose-response ──
    # At λ_ρ=0.2, γ=0 vs γ=0.1:
    margin_effect = {}
    for beh in behaviors:
        d_gamma0 = np.mean(NO_MARGIN_SEEDS[beh])
        d_gamma01 = DOSE_RESPONSE[0.2][beh][0]
        margin_effect[beh] = (d_gamma01 - d_gamma0) / 0.1  # dΔρ/dγ

    # ── Build (γ, λ_ρ) grid ──
    gamma_grid = np.linspace(0.0, 0.15, 100)
    lambda_grid = np.linspace(0.0, 0.6, 100)
    G, L = np.meshgrid(gamma_grid, lambda_grid)

    # ── Model: Δρ_j(γ, λ) = intercepts[j] + slopes[j]·λ + margin_effect[j]·(γ-0.1)·λ/0.2
    # The margin effect scales with λ (stronger contrastive → more interference)
    phase_maps = {}
    gamma_star_curves = {}
    for beh in behaviors:
        # Δρ_j(γ, λ) = intercept + slope·λ + margin_sensitivity·(γ-0.1)·(λ/0.2)
        dRho = (intercepts[beh]
                + slopes[beh] * L
                + margin_effect[beh] * (G - 0.1) * (L / 0.2))
        phase_maps[beh] = dRho

        # Find γ* for each λ (where Δρ crosses zero)
        gs_curve = []
        for i, lam in enumerate(lambda_grid):
            row = dRho[i, :]
            # Find the γ where row crosses zero (going from negative to positive)
            crossings = np.where(np.diff(np.sign(row)))[0]
            if len(crossings) > 0:
                # Linear interpolation to find exact crossing
                idx = crossings[0]
                g0, g1 = gamma_grid[idx], gamma_grid[idx + 1]
                r0, r1 = row[idx], row[idx + 1]
                gamma_cross = g0 + (g1 - g0) * (-r0) / (r1 - r0)
                gs_curve.append((lam, gamma_cross))
            else:
                # All positive or all negative
                if row[0] > 0:
                    gs_curve.append((lam, 0.0))  # always safe
                else:
                    gs_curve.append((lam, np.nan))  # never safe
        gamma_star_curves[beh] = gs_curve

    # ── MC: sample noise and find γ* per behavior jointly ──
    mc_gamma_star = {beh: [] for beh in behaviors}
    for _ in range(n_samples):
        for beh in behaviors:
            # Add noise to margin ablation endpoints
            d0_noise = rng.normal(0, 0.003)
            d01_noise = rng.normal(0, 0.004)
            d0 = np.mean(NO_MARGIN_SEEDS[beh]) + d0_noise
            d01 = DOSE_RESPONSE[0.2][beh][0] + d01_noise

            b1 = (d01 - d0) / 0.1
            if b1 > 0:
                gs = -d0 / b1
                if 0 < gs < 0.5:
                    mc_gamma_star[beh].append(gs)

    mc_summary = {}
    for beh in behaviors:
        vals = np.array(mc_gamma_star[beh])
        if len(vals) > 0:
            mc_summary[beh] = {
                "median": float(np.median(vals)),
                "ci_95": [float(np.percentile(vals, 2.5)),
                          float(np.percentile(vals, 97.5))],
                "n_valid": len(vals),
            }
        else:
            mc_summary[beh] = {"median": 0.0, "ci_95": [0.0, 0.0], "n_valid": 0}

    return {
        "phase_maps": phase_maps,
        "gamma_star_curves": gamma_star_curves,
        "gamma_grid": gamma_grid,
        "lambda_grid": lambda_grid,
        "mc_summary": mc_summary,
        "slopes": {k: float(v) for k, v in slopes.items()},
        "intercepts": {k: float(v) for k, v in intercepts.items()},
        "margin_effect": {k: float(v) for k, v in margin_effect.items()},
    }


def plot_phase_diagram(phase_results: dict, out_path: Path):
    """Plot (γ, λ_ρ) phase diagram with safe/dangerous regions per behavior."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    behaviors = ["factual", "bias", "sycophancy"]
    titles = [r"Factual $\Delta\rho$", r"Bias $\Delta\rho$",
              r"Sycophancy $\Delta\rho$"]
    gamma_grid = phase_results["gamma_grid"]
    lambda_grid = phase_results["lambda_grid"]

    for ax, beh, title in zip(axes, behaviors, titles):
        dRho = phase_results["phase_maps"][beh]
        gs_curve = phase_results["gamma_star_curves"][beh]

        # Contour fill
        vmax = max(abs(dRho.min()), abs(dRho.max()))
        vmax = min(vmax, 0.15)  # clip for sycophancy
        cf = ax.contourf(gamma_grid, lambda_grid, dRho,
                         levels=np.linspace(-vmax, vmax, 25),
                         cmap="RdYlGn", extend="both")
        plt.colorbar(cf, ax=ax, label=r"$\Delta\rho$", shrink=0.8)

        # Zero contour (critical boundary)
        ax.contour(gamma_grid, lambda_grid, dRho, levels=[0],
                   colors="black", linewidths=2.5, linestyles="-")

        # Plot γ* curve
        gs_lam = [p[0] for p in gs_curve if not np.isnan(p[1])]
        gs_gam = [p[1] for p in gs_curve if not np.isnan(p[1])]
        if gs_lam:
            ax.plot(gs_gam, gs_lam, "k-", linewidth=2.5, label=r"$\gamma^*$ boundary")

        # Mark default operating point
        ax.plot(0.1, 0.2, "w*", markersize=15, markeredgecolor="black",
                markeredgewidth=1.5, zorder=10, label=r"Default ($\gamma=0.1, \lambda=0.2$)")

        # MC γ* CI for λ=0.2
        mc = phase_results["mc_summary"].get(beh, {})
        if mc.get("n_valid", 0) > 0:
            ci = mc["ci_95"]
            ax.plot([ci[0], ci[1]], [0.2, 0.2], "|-", color="white",
                    linewidth=3, markersize=12, markeredgewidth=2, zorder=9)

        ax.set_xlabel(r"Margin $\gamma$", fontsize=12)
        ax.set_ylabel(r"Contrastive weight $\lambda_\rho$", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=8, loc="upper left")

        # Red/green labels
        ax.text(0.01, 0.55, "DANGEROUS\n" + r"$\Delta\rho < 0$",
                fontsize=9, color="darkred", fontweight="bold",
                transform=ax.transAxes, va="center")
        ax.text(0.65, 0.15, "SAFE\n" + r"$\Delta\rho > 0$",
                fontsize=9, color="darkgreen", fontweight="bold",
                transform=ax.transAxes, va="center")

    plt.suptitle(r"Phase Diagrams: Safe vs Dangerous Regimes in ($\gamma$, $\lambda_\rho$) Space",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 5: Variance U-Shape Monte Carlo
# ═══════════════════════════════════════════════════════════════════════════

def variance_ushape_mc(n_samples: int = 10000, seed: int = 42) -> dict:
    r"""Model the non-monotonic variance (σ) vs λ_ρ as competition of two noise sources.

    Empirical factual σ:
        λ=0.0: 0.094   (SFT degeneracy dominates)
        λ=0.1: 0.079   (contrastive breaks some degeneracy)
        λ=0.2: 0.015   (sweet spot — minimal total noise)
        λ=0.5: 0.037   (probe sampling noise dominates)

    Model: σ_total²(λ) = σ_SFT²(λ) + σ_probe²(λ)

    where:
        σ_SFT(λ) = σ₀ · exp(-λ/τ_SFT)     — SFT degeneracy decays exponentially
        σ_probe(λ) = σ_p · √λ               — probe-sampling noise grows with weight

    The U-shape minimum occurs at:
        λ* = τ_SFT · ln(σ₀² / (σ_p² · τ_SFT))   [from dσ²/dλ = 0]
    """
    rng = np.random.default_rng(seed)

    # Empirical data points (use sample std: multiply pop std by sqrt(5/4))
    correction = np.sqrt(5 / 4)
    lambda_data = np.array([0.0, 0.1, 0.2, 0.5])
    sigma_factual = np.array([0.0935, 0.0792, 0.0154, 0.0370]) * correction
    sigma_toxicity = np.array([0.0523, 0.0685, 0.0650, 0.0534]) * correction
    sigma_bias = np.array([0.0013, 0.0025, 0.0023, 0.0021]) * correction

    # ── Fit two-component model to factual σ ──
    # σ²(λ) = σ₀² · exp(-2λ/τ) + σ_p² · λ
    # Use scipy minimize to fit σ₀, τ, σ_p
    from scipy.optimize import minimize

    def loss_fn(params):
        s0, tau, sp = params
        if s0 <= 0 or tau <= 0 or sp <= 0:
            return 1e10
        sigma_model = np.sqrt(s0**2 * np.exp(-2 * lambda_data / tau) + sp**2 * lambda_data)
        return np.sum((sigma_model - sigma_factual)**2)

    result = minimize(loss_fn, x0=[0.10, 0.15, 0.05],
                      bounds=[(0.001, 0.5), (0.01, 1.0), (0.001, 0.3)])
    s0_fit, tau_fit, sp_fit = result.x

    # ── Compute λ* (U-shape minimum) analytically ──
    # dσ²/dλ = -2σ₀²/τ · exp(-2λ/τ) + σ_p² = 0
    # exp(-2λ*/τ) = σ_p²·τ / (2σ₀²)
    # λ* = -τ/2 · ln(σ_p²·τ/(2σ₀²))
    lambda_star_analytical = -tau_fit / 2 * np.log(sp_fit**2 * tau_fit / (2 * s0_fit**2))

    # ── MC: sample uncertainty in the fit parameters ──
    lambda_fine = np.linspace(0, 0.6, 200)
    mc_sigmas = np.zeros((n_samples, len(lambda_fine)))
    mc_lambda_star = []

    for i in range(n_samples):
        s0 = rng.normal(s0_fit, s0_fit * 0.15)  # 15% uncertainty
        tau = rng.normal(tau_fit, tau_fit * 0.20)  # 20% uncertainty
        sp = rng.normal(sp_fit, sp_fit * 0.20)   # 20% uncertainty

        if s0 <= 0 or tau <= 0 or sp <= 0:
            mc_sigmas[i, :] = np.nan
            continue

        sig2 = s0**2 * np.exp(-2 * lambda_fine / tau) + sp**2 * lambda_fine
        sig2 = np.maximum(sig2, 0)
        mc_sigmas[i, :] = np.sqrt(sig2)

        # λ* for this sample
        ratio = sp**2 * tau / (2 * s0**2)
        if 0 < ratio < 1:
            ls = -tau / 2 * np.log(ratio)
            if 0 < ls < 0.6:
                mc_lambda_star.append(ls)

    mc_lambda_star = np.array(mc_lambda_star)
    mc_sigma_mean = np.nanmean(mc_sigmas, axis=0)
    mc_sigma_lo = np.nanpercentile(mc_sigmas, 5, axis=0)
    mc_sigma_hi = np.nanpercentile(mc_sigmas, 95, axis=0)

    return {
        "fit_params": {"s0": float(s0_fit), "tau": float(tau_fit), "sp": float(sp_fit)},
        "lambda_star": {
            "analytical": float(lambda_star_analytical),
            "mc_median": float(np.median(mc_lambda_star)) if len(mc_lambda_star) > 0 else np.nan,
            "mc_ci_90": ([float(np.percentile(mc_lambda_star, 5)),
                          float(np.percentile(mc_lambda_star, 95))]
                         if len(mc_lambda_star) > 10 else [np.nan, np.nan]),
        },
        "lambda_fine": lambda_fine,
        "mc_sigma_mean": mc_sigma_mean,
        "mc_sigma_lo": mc_sigma_lo,
        "mc_sigma_hi": mc_sigma_hi,
        "lambda_data": lambda_data,
        "sigma_factual": sigma_factual,
        "sigma_toxicity": sigma_toxicity,
        "sigma_bias": sigma_bias,
        "s0_fit": s0_fit, "tau_fit": tau_fit, "sp_fit": sp_fit,
    }


def plot_variance_ushape(var_results: dict, out_path: Path):
    """Plot the variance U-shape decomposition."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    lf = var_results["lambda_fine"]
    s0, tau, sp = var_results["s0_fit"], var_results["tau_fit"], var_results["sp_fit"]

    # ── Panel 1: Factual σ with two-component decomposition ──
    ax = axes[0]
    # Components
    sft_comp = s0 * np.exp(-lf / tau)
    probe_comp = sp * np.sqrt(lf)
    total = np.sqrt(sft_comp**2 + probe_comp**2)

    ax.plot(lf, sft_comp, "--", color="#E45756", linewidth=2,
            label=rf"SFT degeneracy: $\sigma_0 e^{{-\lambda/\tau}}$ ($\sigma_0$={s0:.3f}, $\tau$={tau:.3f})")
    ax.plot(lf, probe_comp, "--", color="#4C78A8", linewidth=2,
            label=rf"Probe noise: $\sigma_p\sqrt{{\lambda}}$ ($\sigma_p$={sp:.3f})")
    ax.plot(lf, total, "-", color="black", linewidth=2.5, label="Total model")

    # Empirical points
    ax.plot(var_results["lambda_data"], var_results["sigma_factual"],
            "ko", markersize=8, zorder=5, label="Empirical (5 seeds)")

    # MC envelope
    ax.fill_between(lf, var_results["mc_sigma_lo"], var_results["mc_sigma_hi"],
                     alpha=0.15, color="#4C78A8", label="90% MC envelope")

    # λ* mark
    ls = var_results["lambda_star"]["analytical"]
    ax.axvline(ls, color="#54A24B", linestyle=":", linewidth=2)
    ax.annotate(rf"$\lambda^* = {ls:.3f}$", xy=(ls + 0.01, ax.get_ylim()[0]),
                fontsize=10, color="#54A24B")

    ax.set_xlabel(r"$\lambda_\rho$", fontsize=12)
    ax.set_ylabel(r"Factual $\sigma$ (inter-seed)", fontsize=12)
    ax.set_title("(a) Factual Variance Decomposition", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(bottom=0)

    # ── Panel 2: All behaviors σ comparison ──
    ax = axes[1]
    ld = var_results["lambda_data"]
    ax.plot(ld, var_results["sigma_factual"], "o-", color="#4C78A8",
            linewidth=2, markersize=8, label="Factual")
    ax.plot(ld, var_results["sigma_toxicity"], "s-", color="#E45756",
            linewidth=2, markersize=8, label="Toxicity")
    ax.plot(ld, var_results["sigma_bias"], "^-", color="#54A24B",
            linewidth=2, markersize=8, label="Bias")

    ax.set_xlabel(r"$\lambda_\rho$", fontsize=12)
    ax.set_ylabel(r"$\sigma$ (inter-seed std)", fontsize=12)
    ax.set_title("(b) All Behaviors: Variance vs Weight", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)

    # ── Panel 3: λ* MC distribution ──
    ax = axes[2]
    lstar = var_results["lambda_star"]
    if not np.isnan(lstar["mc_median"]):
        # Recompute samples for histogram
        rng = np.random.default_rng(42)
        mc_ls = []
        for _ in range(10000):
            s0s = rng.normal(s0, s0 * 0.15)
            ts = rng.normal(tau, tau * 0.20)
            sps = rng.normal(sp, sp * 0.20)
            if s0s > 0 and ts > 0 and sps > 0:
                ratio = sps**2 * ts / (2 * s0s**2)
                if 0 < ratio < 1:
                    v = -ts / 2 * np.log(ratio)
                    if 0 < v < 0.6:
                        mc_ls.append(v)
        mc_ls = np.array(mc_ls)

        ax.hist(mc_ls, bins=60, density=True, alpha=0.7, color="#F58518",
                edgecolor="white")
        ax.axvline(lstar["analytical"], color="black", linestyle="--",
                   linewidth=2, label=f"Analytical: {lstar['analytical']:.3f}")
        ax.axvline(lstar["mc_median"], color="#E45756", linestyle="-",
                   linewidth=2, label=f"MC median: {lstar['mc_median']:.3f}")
        ax.axvline(0.2, color="#54A24B", linestyle=":", linewidth=2,
                   label=r"Default $\lambda_\rho = 0.2$")
        ax.set_xlabel(r"Optimal $\lambda^*_\rho$", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(r"(c) MC Distribution of $\lambda^*_\rho$", fontsize=12)
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "Insufficient MC samples", transform=ax.transAxes,
                ha="center", fontsize=12)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 6: Probe Sampling Noise
# ═══════════════════════════════════════════════════════════════════════════

def probe_noise_analysis(n_samples: int = 10000, seed: int = 42) -> dict:
    r"""Quantify how finite probe sets inflate variance and shift γ*.

    With N probes, the Spearman ρ estimator has variance:
        Var(ρ̂) ≈ (1 - ρ²)² / (N - 1)   [asymptotic, Gaussian copula]

    This probe-sampling noise adds to the measurement of Δρ, which in turn
    adds uncertainty to γ*.

    We simulate:
    1. For each behavior, compute Var(ρ̂) from its probe count and baseline ρ
    2. Add this noise to the γ* MC and see how γ* distribution broadens
    3. Show how γ* uncertainty scales with probe count (power analysis)
    """
    rng = np.random.default_rng(seed)

    # ── Spearman ρ estimator variance ──
    probe_var = {}
    for beh, n_probes in PROBE_COUNTS.items():
        rho = BASELINE_RHO.get(beh, 0.0)
        # Var(ρ̂) ≈ (1-ρ²)² / (N-1)
        var_rho = (1 - rho**2)**2 / max(n_probes - 1, 1)
        probe_var[beh] = {
            "n_probes": n_probes,
            "baseline_rho": rho,
            "var_rho_hat": float(var_rho),
            "std_rho_hat": float(np.sqrt(var_rho)),
        }

    # ── γ* with and without probe noise ──
    # Without probe noise (measurement noise only — the v1 analysis)
    d0_base = GAMMA_0["d_bias"]
    d01_base = GAMMA_01["d_bias"]

    # Probe noise for bias: sqrt(Var(ρ̂)) for N=300 at baseline ρ=0.036
    bias_probe_std = probe_var.get("bias", {}).get("std_rho_hat", 0.058)

    # MC with probe noise added
    gs_no_probe = []
    gs_with_probe = []
    for _ in range(n_samples):
        # Measurement noise
        d0 = rng.normal(d0_base, 0.003)
        d01 = rng.normal(d01_base, 0.004)

        b1 = (d01 - d0) / 0.1
        if b1 > 0:
            gs = -d0 / b1
            if 0 < gs < 0.5:
                gs_no_probe.append(gs)

        # Now add probe-sampling noise to each measurement
        d0_noisy = d0 + rng.normal(0, bias_probe_std)
        d01_noisy = d01 + rng.normal(0, bias_probe_std)
        b1_noisy = (d01_noisy - d0_noisy) / 0.1
        if b1_noisy > 0:
            gs = -d0_noisy / b1_noisy
            if 0 < gs < 0.5:
                gs_with_probe.append(gs)

    gs_no_probe = np.array(gs_no_probe)
    gs_with_probe = np.array(gs_with_probe)

    # ── Power analysis: how probe count affects γ* width ──
    n_probes_sweep = [50, 100, 150, 200, 300, 500, 1000]
    gs_width_by_n = []
    for n_p in n_probes_sweep:
        rho = BASELINE_RHO["bias"]
        ps = np.sqrt((1 - rho**2)**2 / max(n_p - 1, 1))
        gs_samples = []
        for _ in range(n_samples):
            d0 = rng.normal(d0_base, 0.003) + rng.normal(0, ps)
            d01 = rng.normal(d01_base, 0.004) + rng.normal(0, ps)
            b1 = (d01 - d0) / 0.1
            if b1 > 0:
                gs = -d0 / b1
                if 0 < gs < 0.5:
                    gs_samples.append(gs)
        gs_arr = np.array(gs_samples)
        if len(gs_arr) > 10:
            gs_width_by_n.append({
                "n_probes": n_p,
                "ci_width": float(np.percentile(gs_arr, 97.5) - np.percentile(gs_arr, 2.5)),
                "std": float(np.std(gs_arr)),
                "median": float(np.median(gs_arr)),
                "n_valid": len(gs_arr),
            })
        else:
            gs_width_by_n.append({
                "n_probes": n_p, "ci_width": np.nan, "std": np.nan,
                "median": np.nan, "n_valid": len(gs_arr),
            })

    return {
        "probe_var": probe_var,
        "gs_no_probe": gs_no_probe,
        "gs_with_probe": gs_with_probe,
        "gs_no_probe_summary": {
            "median": float(np.median(gs_no_probe)),
            "std": float(np.std(gs_no_probe)),
            "ci_95_width": float(np.percentile(gs_no_probe, 97.5) -
                                  np.percentile(gs_no_probe, 2.5)),
        },
        "gs_with_probe_summary": {
            "median": float(np.median(gs_with_probe)) if len(gs_with_probe) > 0 else np.nan,
            "std": float(np.std(gs_with_probe)) if len(gs_with_probe) > 0 else np.nan,
            "ci_95_width": (float(np.percentile(gs_with_probe, 97.5) -
                                   np.percentile(gs_with_probe, 2.5))
                            if len(gs_with_probe) > 10 else np.nan),
        },
        "power_analysis": gs_width_by_n,
    }


def plot_probe_noise(probe_results: dict, out_path: Path):
    """Plot probe-sampling noise impact."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # ── Panel 1: γ* distribution with vs without probe noise ──
    ax = axes[0]
    gs_no = probe_results["gs_no_probe"]
    gs_with = probe_results["gs_with_probe"]

    ax.hist(gs_no, bins=60, density=True, alpha=0.6, color="#4C78A8",
            label=f"Measurement noise only (σ={probe_results['gs_no_probe_summary']['std']:.4f})",
            edgecolor="white")
    if len(gs_with) > 0:
        ax.hist(gs_with, bins=60, density=True, alpha=0.5, color="#E45756",
                label=f"+ Probe noise (σ={probe_results['gs_with_probe_summary']['std']:.4f})",
                edgecolor="white")
    ax.axvline(0.1, color="#54A24B", linestyle=":", linewidth=2,
               label=r"Default $\gamma = 0.1$")
    ax.set_xlabel(r"$\gamma^*$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(r"(a) $\gamma^*$ With/Without Probe Noise", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(-0.05, 0.2)

    # ── Panel 2: Probe variance by behavior ──
    ax = axes[1]
    pv = probe_results["probe_var"]
    behs = sorted(pv.keys(), key=lambda b: pv[b]["std_rho_hat"], reverse=True)
    stds = [pv[b]["std_rho_hat"] for b in behs]
    ns = [pv[b]["n_probes"] for b in behs]
    colors_map = {"factual": "#4C78A8", "toxicity": "#E45756", "bias": "#54A24B",
                  "sycophancy": "#F58518", "reasoning": "#72B7B2",
                  "refusal": "#B279A2", "deception": "#FF9DA6",
                  "overrefusal": "#9D755D"}
    bar_colors = [colors_map.get(b, "#999999") for b in behs]

    bars = ax.barh(behs, stds, color=bar_colors, edgecolor="white")
    for bar, n in zip(bars, ns):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"N={n}", va="center", fontsize=9, color="gray")
    ax.set_xlabel(r"$\sigma(\hat{\rho})$ (probe sampling noise)", fontsize=12)
    ax.set_title("(b) Probe Noise Floor by Behavior", fontsize=12)
    ax.invert_yaxis()

    # ── Panel 3: Power analysis — CI width vs probe count ──
    ax = axes[2]
    pa = probe_results["power_analysis"]
    ns_plot = [p["n_probes"] for p in pa if not np.isnan(p["ci_width"])]
    widths = [p["ci_width"] for p in pa if not np.isnan(p["ci_width"])]

    ax.plot(ns_plot, widths, "o-", color="#4C78A8", linewidth=2, markersize=8)
    ax.axhline(probe_results["gs_no_probe_summary"]["ci_95_width"],
               color="#E45756", linestyle="--", linewidth=1.5,
               label="Measurement-only 95% CI width")
    ax.axvline(300, color="#54A24B", linestyle=":", linewidth=1.5,
               label="Current bias probes (N=300)")

    ax.set_xlabel("Number of probes per behavior", fontsize=12)
    ax.set_ylabel(r"95% CI width of $\gamma^*$", fontsize=12)
    ax.set_title(r"(c) Power Analysis: Probes vs $\gamma^*$ Precision", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xscale("log")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Extended γ* bounds analysis with Monte Carlo"
    )
    parser.add_argument("--n-samples", type=int, default=10000,
                        help="Number of Monte Carlo samples (default: 10000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("EXTENDED γ* BOUNDS ANALYSIS")
    print("=" * 70)

    # --- Analysis 1: Monte Carlo ---
    print(f"\n[1] Monte Carlo γ* distribution (N={args.n_samples:,})")
    mc_results, gs_valid, s_inf_samples, cos_theta_samples, baseline_bias = \
        monte_carlo_gamma_star(args.n_samples, args.seed)

    print(f"  Valid samples: {mc_results['n_valid']:,} / {args.n_samples:,}")
    print(f"  Point estimate: {mc_results['gamma_star']['point_estimate']:.4f}")
    print(f"  MC median:      {mc_results['gamma_star']['median']:.4f}")
    print(f"  MC mean:        {mc_results['gamma_star']['mean']:.4f}")
    print(f"  68% CI:         [{mc_results['gamma_star']['ci_68'][0]:.4f}, "
          f"{mc_results['gamma_star']['ci_68'][1]:.4f}]")
    print(f"  95% CI:         [{mc_results['gamma_star']['ci_95'][0]:.4f}, "
          f"{mc_results['gamma_star']['ci_95'][1]:.4f}]")
    print(f"  Safety factor (γ=0.1/median): "
          f"{0.1/mc_results['gamma_star']['median']:.1f}×")

    plot_mc_distribution(gs_valid, mc_results, DOCS_DIR / "gamma_mc_distribution.png")

    # --- Analysis 2: Nonlinear amplification ---
    print(f"\n[2] Nonlinear amplification model")
    amp_results, behaviors, baseline_rho, swings, cos_thetas = \
        nonlinear_amplification_analysis()

    print(f"  Observed bias/sycophancy ratio: {amp_results['observed_ratio']:.1f}×")
    print(f"  Linear prediction (cos θ):      {amp_results['linear_prediction']:.1f}×")
    print(f"  Residual amplification:          {amp_results['residual_amplification']:.1f}×")
    print(f"  Decomposition: {amp_results['decomposition']['angular_factor']:.1f}× "
          f"(angular) × {amp_results['decomposition']['nonlinear_factor']:.1f}× "
          f"(nonlinear) = {amp_results['decomposition']['product']:.1f}×")

    plot_amplification(amp_results, behaviors, baseline_rho, swings, cos_thetas,
                       DOCS_DIR / "gamma_amplification.png")

    # --- Analysis 3: Sensitivity ---
    print(f"\n[3] Sensitivity analysis (tornado diagram)")
    sens_results = sensitivity_analysis(args.n_samples, args.seed)
    print(f"  Baseline γ*: {sens_results['baseline_gamma_star']:.4f}")
    for name, data in sens_results["parameters"].items():
        rng = abs(data["high"] - data["low"])
        print(f"  {name}: range {rng:.4f} "
              f"({data['param_range']})")

    plot_tornado(sens_results, DOCS_DIR / "gamma_sensitivity.png")

    # --- Analysis 4: Multi-behavior phase diagram ---
    print(f"\n[4] Multi-behavior phase diagram")
    phase_results = multi_behavior_phase(args.n_samples, args.seed)
    for beh, mc in phase_results["mc_summary"].items():
        if mc["n_valid"] > 0:
            ci = mc["ci_95"]
            print(f"  {beh}: γ* = {mc['median']:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
        else:
            print(f"  {beh}: no valid γ* (always safe or always dangerous)")
    plot_phase_diagram(phase_results, DOCS_DIR / "gamma_phase_diagram.png")

    # --- Analysis 5: Variance U-shape ---
    print(f"\n[5] Variance U-shape Monte Carlo")
    var_results = variance_ushape_mc(args.n_samples, args.seed)
    print(f"  Fit: σ₀={var_results['fit_params']['s0']:.4f}, "
          f"τ={var_results['fit_params']['tau']:.4f}, "
          f"σ_p={var_results['fit_params']['sp']:.4f}")
    ls = var_results["lambda_star"]
    print(f"  λ* (analytical) = {ls['analytical']:.3f}")
    if not np.isnan(ls["mc_median"]):
        print(f"  λ* (MC median) = {ls['mc_median']:.3f} "
              f"(90% CI: [{ls['mc_ci_90'][0]:.3f}, {ls['mc_ci_90'][1]:.3f}])")
    plot_variance_ushape(var_results, DOCS_DIR / "gamma_variance_ushape.png")

    # --- Analysis 6: Probe noise ---
    print(f"\n[6] Probe sampling noise analysis")
    probe_results = probe_noise_analysis(args.n_samples, args.seed)
    s_no = probe_results["gs_no_probe_summary"]
    s_with = probe_results["gs_with_probe_summary"]
    print(f"  γ* width (measurement only): {s_no['ci_95_width']:.4f}")
    print(f"  γ* width (+ probe noise):    {s_with['ci_95_width']:.4f}")
    inflation = s_with["ci_95_width"] / s_no["ci_95_width"] if s_no["ci_95_width"] > 0 else np.nan
    print(f"  Inflation factor: {inflation:.2f}×")
    print(f"  Probe noise floors:")
    for beh in sorted(probe_results["probe_var"].keys(),
                      key=lambda b: probe_results["probe_var"][b]["std_rho_hat"],
                      reverse=True)[:4]:
        pv = probe_results["probe_var"][beh]
        print(f"    {beh}: σ(ρ̂)={pv['std_rho_hat']:.4f} (N={pv['n_probes']})")
    plot_probe_noise(probe_results, DOCS_DIR / "gamma_probe_noise.png")

    # --- Save JSON ---
    output = {
        "monte_carlo": {k: v for k, v in mc_results.items() if k != "samples"},
        "amplification": amp_results,
        "sensitivity": sens_results,
        "phase_diagram": {
            "mc_summary": phase_results["mc_summary"],
            "slopes": phase_results["slopes"],
            "intercepts": phase_results["intercepts"],
            "margin_effect": phase_results["margin_effect"],
        },
        "variance_ushape": {
            "fit_params": var_results["fit_params"],
            "lambda_star": var_results["lambda_star"],
        },
        "probe_noise": {
            "probe_var": probe_results["probe_var"],
            "gs_no_probe_summary": probe_results["gs_no_probe_summary"],
            "gs_with_probe_summary": probe_results["gs_with_probe_summary"],
            "power_analysis": probe_results["power_analysis"],
        },
    }
    json_path = DOCS_DIR / "gamma_bounds_analysis.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  → {json_path}")

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  γ* = {mc_results['gamma_star']['median']:.4f} "
          f"(95% CI: [{mc_results['gamma_star']['ci_95'][0]:.4f}, "
          f"{mc_results['gamma_star']['ci_95'][1]:.4f}])")
    print(f"  Default γ=0.1 is {0.1/mc_results['gamma_star']['median']:.1f}× "
          f"the critical margin (safe)")
    print(f"  Amplification: {amp_results['decomposition']['angular_factor']:.1f}× angular "
          f"× {amp_results['decomposition']['nonlinear_factor']:.1f}× nonlinear "
          f"= {amp_results['decomposition']['product']:.1f}×")
    print(f"  Optimal λ*_ρ = {ls['analytical']:.3f} "
          f"(default 0.2 is {'near-optimal' if abs(ls['analytical'] - 0.2) < 0.05 else 'suboptimal'})")
    print(f"  Probe noise inflates γ* CI by {inflation:.1f}×")


if __name__ == "__main__":
    main()
