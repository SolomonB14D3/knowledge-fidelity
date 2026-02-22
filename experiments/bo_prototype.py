#!/usr/bin/env python3
"""
Bayesian Optimization Prototype for Behavioral Compression Tuning

Proof-of-concept: fit a Gaussian Process surrogate on existing sweep data,
then use acquisition functions to suggest the next best (ratio, freeze_ratio)
configs to evaluate — optimizing for a target behavioral rho.

This prototype:
  1. Loads all existing data (joint ablation + cross-behavioral sweep)
  2. Fits a GP per behavior
  3. Optimizes acquisition (Expected Improvement) for a target behavior
  4. Predicts optimal config + suggests next eval points
  5. Generates a matplotlib surface plot of the GP predictions

Uses only scipy + sklearn (no skopt/Ax dependency).

Usage:
    python experiments/bo_prototype.py
    python experiments/bo_prototype.py --target bias
    python experiments/bo_prototype.py --target sycophancy --secondary factual=-0.01
    python experiments/bo_prototype.py --plot
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

warnings.filterwarnings("ignore", category=UserWarning)

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_all_data():
    """Load all available sweep data into a unified format.

    Returns:
        dict: {behavior: [(ratio, freeze_ratio, model_idx, rho_delta), ...]}
        Model index: 0=Qwen-0.5B, 1=Qwen-7B, 2=Mistral-7B
    """
    data = {}

    # ── Joint ablation data (ratio sweep, freeze=0.75 fixed via CF90) ─────
    ablation_files = [
        ("joint_ablation_Qwen2.5-0.5B.json", 0, "qwen-0.5b"),
        ("joint_ablation_Qwen2.5-7B-Instruct.json", 1, "qwen-7b"),
    ]

    behavior_map = {
        "default": "factual",
        "mandela": "mandela",
        "medical": "medical",
    }

    for fname, model_idx, model_name in ablation_files:
        path = RESULTS_DIR / fname
        if not path.exists():
            continue
        with open(path) as f:
            entries = json.load(f)
        for entry in entries:
            ratio = entry["ratio"]
            if ratio >= 1.0:
                continue  # skip baseline (delta=0 by definition)
            freeze = 0.75  # CF90 default freeze ratio
            for prefix, behavior in behavior_map.items():
                before = entry.get(f"{prefix}_rho_before", 0)
                after = entry.get(f"{prefix}_rho_after", 0)
                delta = after - before
                data.setdefault(behavior, []).append(
                    (ratio, freeze, model_idx, delta, model_name)
                )

    # ── Cross-behavioral sweep (ratio sweep, freeze=0.75 fixed) ──────────
    sweep_path = RESULTS_DIR / "cross_behavioral" / "sweep.json"
    if sweep_path.exists():
        with open(sweep_path) as f:
            sweep = json.load(f)

        for key, entry in sweep.items():
            if "_baseline" in key:
                continue
            ratio = entry.get("ratio", 0)
            if ratio >= 1.0:
                continue
            freeze = 0.75  # sweep uses CF90 freeze
            model_idx = 1  # Qwen-7B

            for behavior, bdata in entry.get("behaviors", {}).items():
                if isinstance(bdata, dict) and "delta" in bdata:
                    data.setdefault(behavior, []).append(
                        (ratio, freeze, model_idx, bdata["delta"], "qwen-7b")
                    )

    # ── Freeze-ratio sweep data (if exists) ───────────────────────────────
    freeze_sweep_path = RESULTS_DIR / "freeze_sweep" / "sweep.json"
    if freeze_sweep_path.exists():
        with open(freeze_sweep_path) as f:
            freeze_sweep = json.load(f)
        for key, entry in freeze_sweep.items():
            if "_baseline" in key:
                continue
            ratio = entry.get("compress_ratio", 0.7)
            freeze = entry.get("freeze_ratio", 0.75)
            model_name = entry.get("model", "unknown")
            model_idx = {"qwen2.5-7b": 1, "llama3.1-8b": 3}.get(model_name, 1)
            for behavior, bdata in entry.get("behaviors", {}).items():
                if isinstance(bdata, dict) and "delta" in bdata:
                    data.setdefault(behavior, []).append(
                        (ratio, freeze, model_idx, bdata["delta"], model_name)
                    )

    return data


def summarize_data(data):
    """Print summary of loaded data."""
    print("=" * 70)
    print("LOADED DATA SUMMARY")
    print("=" * 70)
    total = 0
    for behavior in sorted(data):
        points = data[behavior]
        n = len(points)
        total += n
        deltas = [p[3] for p in points]
        ratios = sorted(set(p[0] for p in points))
        models = sorted(set(p[4] for p in points))
        print(f"  {behavior:<14}: {n:>3} points | "
              f"delta range [{min(deltas):+.4f}, {max(deltas):+.4f}] | "
              f"ratios: {[f'{r:.0%}' for r in ratios]} | "
              f"models: {models}")
    print(f"\n  TOTAL: {total} data points across {len(data)} behaviors")
    return total


# ═══════════════════════════════════════════════════════════════════════════
# GAUSSIAN PROCESS SURROGATE
# ═══════════════════════════════════════════════════════════════════════════

def fit_gp(data_points, use_model_feature=False):
    """Fit a GP to data points for one behavior.

    Args:
        data_points: list of (ratio, freeze_ratio, model_idx, delta, model_name)
        use_model_feature: if True, include model_idx as a feature

    Returns:
        (gp, X, y) — fitted GP, input array, delta array
    """
    if use_model_feature:
        X = np.array([[p[0], p[1], p[2]] for p in data_points])
    else:
        X = np.array([[p[0], p[1]] for p in data_points])
    y = np.array([p[3] for p in data_points])

    # Matern 5/2 kernel — smooth but flexible, good for small data
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e2)) *
        Matern(length_scale=np.ones(X.shape[1]) * 0.3,
               length_scale_bounds=(0.05, 2.0), nu=2.5) +
        WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        alpha=1e-6,
        normalize_y=True,
    )
    gp.fit(X, y)
    return gp, X, y


# ═══════════════════════════════════════════════════════════════════════════
# ACQUISITION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def expected_improvement(X_new, gp, y_best, xi=0.01):
    """Expected Improvement acquisition function.

    Args:
        X_new: candidate points (n, d)
        gp: fitted GP
        y_best: best observed value
        xi: exploration-exploitation tradeoff (0.01 = slight exploration)

    Returns:
        EI values (n,)
    """
    mu, sigma = gp.predict(X_new, return_std=True)
    sigma = np.maximum(sigma, 1e-8)
    Z = (mu - y_best - xi) / sigma
    ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei


def optimize_acquisition(gp, y_best, bounds, n_restarts=50, xi=0.01):
    """Find the point that maximizes Expected Improvement.

    Args:
        gp: fitted GP
        y_best: best observed delta so far
        bounds: [(lo, hi), ...] for each dimension
        n_restarts: number of random restarts for optimization
        xi: EI exploration parameter

    Returns:
        (best_x, best_ei) — optimal point and its EI value
    """
    dim = len(bounds)
    best_x = None
    best_ei = -np.inf

    for _ in range(n_restarts):
        x0 = np.array([np.random.uniform(lo, hi) for lo, hi in bounds])

        def neg_ei(x):
            x_2d = x.reshape(1, -1)
            return -expected_improvement(x_2d, gp, y_best, xi=xi)[0]

        result = minimize(neg_ei, x0, bounds=bounds, method="L-BFGS-B")
        if -result.fun > best_ei:
            best_ei = -result.fun
            best_x = result.x

    return best_x, best_ei


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-OBJECTIVE SCALARIZATION
# ═══════════════════════════════════════════════════════════════════════════

def multi_objective_score(gps, x, weights):
    """Compute weighted sum of GP predictions across behaviors.

    Args:
        gps: dict of {behavior: fitted_gp}
        x: input point (ratio, freeze_ratio)
        weights: dict of {behavior: weight} (positive = maximize, negative = penalize)

    Returns:
        (score, uncertainties) — weighted score and per-behavior uncertainty
    """
    x_2d = np.array(x).reshape(1, -1)
    score = 0.0
    uncertainties = {}
    for behavior, gp in gps.items():
        w = weights.get(behavior, 0.0)
        if w == 0:
            continue
        mu, sigma = gp.predict(x_2d, return_std=True)
        score += w * mu[0]
        uncertainties[behavior] = {"mu": float(mu[0]), "sigma": float(sigma[0])}
    return float(score), uncertainties


# ═══════════════════════════════════════════════════════════════════════════
# MAIN OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════

def run_bo(target="bias", secondary=None, n_suggestions=5):
    """Run the full BO prototype.

    Args:
        target: primary behavior to maximize
        secondary: dict of {behavior: min_threshold} constraints
        n_suggestions: number of next points to suggest
    """
    data = load_all_data()
    n_total = summarize_data(data)

    if n_total < 3:
        print("\n⚠ Not enough data points to fit a GP. Need at least 3.")
        return

    # ── Fit GPs per behavior ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"FITTING GP SURROGATES")
    print(f"{'=' * 70}")

    gps = {}
    for behavior in sorted(data):
        points = data[behavior]
        if len(points) < 2:
            print(f"  {behavior}: skipped (only {len(points)} points)")
            continue

        gp, X, y = fit_gp(points)
        gps[behavior] = gp

        # Show fit quality
        y_pred = gp.predict(X)
        residuals = y - y_pred
        rmse = np.sqrt(np.mean(residuals**2))
        r2 = 1 - np.sum(residuals**2) / np.sum((y - y.mean())**2) if len(y) > 1 else 0
        print(f"  {behavior:<14}: {len(points)} pts, RMSE={rmse:.4f}, R²={r2:.3f}")
        print(f"    kernel: {gp.kernel_}")

    if target not in gps:
        print(f"\n⚠ Target behavior '{target}' not in available data: {list(gps.keys())}")
        return

    # ── Optimize for target behavior ──────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"OPTIMIZING FOR: {target.upper()}")
    if secondary:
        print(f"SECONDARY CONSTRAINTS: {secondary}")
    print(f"{'=' * 70}")

    target_gp = gps[target]
    target_data = data[target]
    target_deltas = [p[3] for p in target_data]
    y_best = max(target_deltas)
    best_point = target_data[np.argmax(target_deltas)]

    print(f"\n  Best observed: ratio={best_point[0]:.0%}, freeze={best_point[1]:.0%}, "
          f"delta={y_best:+.4f} ({best_point[4]})")

    # Grid prediction: show GP's view of the landscape
    print(f"\n  --- GP Predicted Landscape (ratio × freeze) ---")
    ratios = np.arange(0.45, 0.96, 0.05)
    freezes = np.array([0.75])  # For now, most data is at freeze=0.75

    # If we have freeze sweep data, expand the grid
    freeze_values = sorted(set(p[1] for p in target_data))
    if len(freeze_values) > 1:
        freezes = np.arange(0.4, 0.96, 0.1)

    if len(freezes) == 1:
        # 1D: just ratio
        print(f"  (freeze fixed at {freezes[0]:.0%} — no freeze sweep data yet)")
        print(f"  {'Ratio':<8} | {'Predicted δ':>12} | {'±σ':>8} | {'EI':>8}")
        print(f"  {'-'*44}")

        X_grid = np.array([[r, freezes[0]] for r in ratios])
        mu, sigma = target_gp.predict(X_grid, return_std=True)
        ei = expected_improvement(X_grid, target_gp, y_best)

        for i, r in enumerate(ratios):
            marker = " ◀ best predicted" if mu[i] == mu.max() else ""
            print(f"  {r:>6.0%}  | {mu[i]:>+10.4f}   | {sigma[i]:>6.4f} | "
                  f"{ei[i]:>6.4f}{marker}")

        # Best predicted
        best_idx = np.argmax(mu)
        print(f"\n  GP optimum: ratio={ratios[best_idx]:.0%}, "
              f"predicted delta={mu[best_idx]:+.4f} ± {sigma[best_idx]:.4f}")
    else:
        # 2D: ratio × freeze
        print(f"  {'':>8}", end="")
        for f in freezes:
            print(f" | f={f:.0%}", end="")
        print()

        for r in ratios:
            row = f"  r={r:.0%}  "
            for f in freezes:
                X_pt = np.array([[r, f]])
                mu, sigma = target_gp.predict(X_pt, return_std=True)
                row += f" | {mu[0]:+.3f}"
            print(row)

    # ── Suggest next evaluation points ────────────────────────────────
    print(f"\n  --- Next {n_suggestions} Suggested Evaluations (by EI) ---")
    bounds = [(0.45, 0.95), (0.4, 0.95)]
    suggestions = []

    for i in range(n_suggestions):
        x_next, ei_next = optimize_acquisition(target_gp, y_best, bounds)
        mu_next, sigma_next = target_gp.predict(x_next.reshape(1, -1), return_std=True)
        suggestions.append({
            "ratio": float(x_next[0]),
            "freeze": float(x_next[1]),
            "predicted_delta": float(mu_next[0]),
            "uncertainty": float(sigma_next[0]),
            "ei": float(ei_next),
        })

        # Add a phantom observation to diversify (Thompson sampling effect)
        X_phantom = x_next.reshape(1, -1)
        y_phantom = np.array([mu_next[0]])  # use GP mean as phantom
        X_all = np.vstack([target_gp.X_train_, X_phantom])
        y_all = np.concatenate([target_gp.y_train_, y_phantom])
        target_gp.fit(X_all, y_all)

    # Sort by EI
    suggestions.sort(key=lambda s: s["ei"], reverse=True)
    for i, s in enumerate(suggestions):
        print(f"  {i+1}. ratio={s['ratio']:.2f}, freeze={s['freeze']:.2f} → "
              f"predicted delta={s['predicted_delta']:+.4f} ± {s['uncertainty']:.4f} "
              f"(EI={s['ei']:.4f})")

    # ── Multi-objective analysis ──────────────────────────────────────
    if len(gps) >= 2:
        print(f"\n  --- Multi-Objective Analysis ---")
        # Default weights: maximize target, penalize factual drop
        weights = {target: 1.0}
        if "factual" in gps and target != "factual":
            weights["factual"] = -0.2  # penalize factual drops
        if secondary:
            for beh, w in secondary.items():
                weights[beh] = w

        print(f"  Weights: {weights}")

        # Grid search for multi-obj optimum
        best_score = -np.inf
        best_config = None
        for r in np.arange(0.45, 0.96, 0.02):
            for f in np.arange(0.4, 0.96, 0.05):
                x = [r, f]
                score, _ = multi_objective_score(gps, x, weights)
                if score > best_score:
                    best_score = score
                    best_config = (r, f)

        if best_config:
            _, details = multi_objective_score(gps, list(best_config), weights)
            print(f"  Multi-obj optimum: ratio={best_config[0]:.2f}, "
                  f"freeze={best_config[1]:.2f} (score={best_score:+.4f})")
            for beh, d in details.items():
                w = weights.get(beh, 0)
                print(f"    {beh}: predicted delta={d['mu']:+.4f} ± {d['sigma']:.4f} "
                      f"(weight={w:+.1f})")

    # ── Save results ──────────────────────────────────────────────────
    output = {
        "target": target,
        "secondary": secondary,
        "n_data_points": n_total,
        "behaviors_available": list(gps.keys()),
        "best_observed": {
            "ratio": float(best_point[0]),
            "freeze": float(best_point[1]),
            "delta": float(y_best),
            "model": best_point[4],
        },
        "suggestions": suggestions,
    }
    out_path = RESULTS_DIR / "bo_prototype_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    return output


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def plot_gp_surfaces(behaviors=None):
    """Plot GP predictions as 1D curves (ratio) with confidence bands."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_all_data()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if behaviors is None:
        behaviors = sorted(data.keys())

    n_behaviors = len(behaviors)
    fig, axes = plt.subplots(1, n_behaviors, figsize=(4 * n_behaviors, 4),
                             squeeze=False, sharey=True)

    ratios_fine = np.linspace(0.45, 0.95, 100)

    for idx, behavior in enumerate(behaviors):
        ax = axes[0, idx]
        points = data.get(behavior, [])
        if len(points) < 2:
            ax.set_title(f"{behavior}\n(insufficient data)")
            continue

        gp, X, y = fit_gp(points)

        # Predict on fine grid (freeze fixed at 0.75)
        X_fine = np.column_stack([ratios_fine, np.full_like(ratios_fine, 0.75)])
        mu, sigma = gp.predict(X_fine, return_std=True)

        # Plot GP mean + confidence band
        ax.plot(ratios_fine, mu, "b-", linewidth=2, label="GP mean")
        ax.fill_between(ratios_fine, mu - 2 * sigma, mu + 2 * sigma,
                        alpha=0.2, color="blue", label="±2σ")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        # Plot observed data
        X_obs = np.array([p[0] for p in points])
        y_obs = np.array([p[3] for p in points])
        models = [p[4] for p in points]
        model_set = sorted(set(models))
        markers = ["o", "s", "^", "D"]
        for m_idx, model in enumerate(model_set):
            mask = np.array([m == model for m in models])
            ax.scatter(X_obs[mask], y_obs[mask],
                       marker=markers[m_idx % len(markers)],
                       s=60, zorder=5, label=model, edgecolors="black", linewidth=0.5)

        ax.set_title(f"{behavior}", fontweight="bold")
        ax.set_xlabel("SVD Ratio")
        if idx == 0:
            ax.set_ylabel("Δρ (rho delta)")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    plt.suptitle("GP Surrogate: Predicted Δρ vs SVD Compression Ratio\n"
                 "(freeze=75%, Matern 5/2 kernel)", fontsize=11, fontweight="bold")
    plt.tight_layout()

    out_path = FIGURES_DIR / "bo_gp_surfaces.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Bayesian Optimization prototype for behavioral compression tuning"
    )
    parser.add_argument("--target", default="bias",
                        help="Primary behavior to maximize (default: bias)")
    parser.add_argument("--secondary", default=None,
                        help="Secondary constraints as behavior=weight pairs, "
                             "comma-separated (e.g., 'factual=-0.01,sycophancy=0.3')")
    parser.add_argument("--n-suggestions", type=int, default=5,
                        help="Number of next points to suggest")
    parser.add_argument("--plot", action="store_true",
                        help="Generate GP surface plots")
    args = parser.parse_args()

    secondary = None
    if args.secondary:
        secondary = {}
        for pair in args.secondary.split(","):
            beh, w = pair.split("=")
            secondary[beh.strip()] = float(w)

    if args.plot:
        plot_gp_surfaces()

    run_bo(target=args.target, secondary=secondary,
           n_suggestions=args.n_suggestions)


if __name__ == "__main__":
    main()
