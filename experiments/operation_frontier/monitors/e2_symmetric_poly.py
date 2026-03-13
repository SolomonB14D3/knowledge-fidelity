"""
Monitor: e2_symmetric_poly_variance

Checks whether e₂ = r12·r13 + r12·r23 + r13·r23 (the second elementary symmetric
polynomial of pairwise distances) is approximately conserved along a trajectory.

Discovery provenance: Phase 2 batch 2, 2026-03-13 (C10).
Oracle FAIL + Checker PASS — margin = -1.67, frac_var = 2.69e-03 on figure-8.
This is the primary Oracle FAIL + Checker PASS candidate: numerically real but
not in the model's training data.

Mechanism: same Z3 choreographic cancellation as e1 (r12+r13+r23). For e2,
the 3ω₀ content is amplified by the cross-product structure, giving ~5× worse
frac_var than e1 but still well within the near-conservation threshold.

Threshold tuning:
  < 5e-3  → strong conservation (figure-8 class, like e2 at 2.69e-03)
  5e-3 to 2e-2 → marginal (e3 at 1.85e-02 barely fails)
  > 2e-2  → not conserved (random/hierarchical baseline ~0.5–0.9)

Usage in problem.yaml:
  monitors:
    - margin_sign
    - e2_symmetric_poly_variance
"""

import numpy as np
from typing import Dict


def compute_e2_symmetric_poly_variance(positions: np.ndarray) -> Dict:
    """
    Check whether e₂ = r12·r13 + r12·r23 + r13·r23 is approximately conserved.

    This is the second elementary symmetric polynomial of the three pairwise
    distances. It vanishes iff two of the three distances vanish simultaneously.

    Args:
        positions: shape (T, 3, D) — T timesteps, 3 bodies, D spatial dims

    Returns dict with:
        e2_mean: mean value of e2 over the trajectory
        frac_var: fractional variation σ / |mean| (lower = more conserved)
        is_conserved: True if frac_var < 5e-3
        vs_e1_ratio: frac_var(e2) / frac_var(e1) — should be ~5 for figure-8
    """
    p1 = positions[:, 0, :]
    p2 = positions[:, 1, :]
    p3 = positions[:, 2, :]
    r12 = np.linalg.norm(p1 - p2, axis=1)
    r13 = np.linalg.norm(p1 - p3, axis=1)
    r23 = np.linalg.norm(p2 - p3, axis=1)

    e2 = r12 * r13 + r12 * r23 + r13 * r23
    e1 = r12 + r13 + r23

    mean_e2 = np.mean(e2)
    frac_var_e2 = np.std(e2) / mean_e2 if mean_e2 > 0 else 0.0

    mean_e1 = np.mean(e1)
    frac_var_e1 = np.std(e1) / mean_e1 if mean_e1 > 0 else 0.0
    vs_e1_ratio = frac_var_e2 / frac_var_e1 if frac_var_e1 > 0 else float("inf")

    return {
        "e2_mean": float(mean_e2),
        "frac_var": float(frac_var_e2),
        "is_conserved": bool(frac_var_e2 < 5e-3),
        "vs_e1_ratio": float(vs_e1_ratio),
        # Also return e1 frac_var for comparison
        "e1_frac_var": float(frac_var_e1),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from conservation_checker import integrate_3body, ic_figure8, ic_general_random, ic_hierarchical

    print("e2_symmetric_poly monitor — self-test")
    print(f"{'IC':20s}  {'e2_frac_var':>12s}  {'e1_frac_var':>12s}  {'ratio':>7s}  {'conserved':>9s}")
    print("-" * 70)

    configs = [
        ("figure-8", ic_figure8()),
        ("random(42)", ic_general_random(42)),
        ("random(17)", ic_general_random(17)),
        ("hierarchical", ic_hierarchical()),
    ]

    for name, (masses, pos0, vel0) in configs:
        t, state = integrate_3body(masses, pos0, vel0, t_end=100, n_points=5000)
        N = 3
        positions = state[:, :3*N].reshape(-1, N, 3)
        r = compute_e2_symmetric_poly_variance(positions)
        print(f"{name:20s}  {r['frac_var']:12.3e}  {r['e1_frac_var']:12.3e}  "
              f"{r['vs_e1_ratio']:7.2f}  {str(r['is_conserved']):>9s}")
