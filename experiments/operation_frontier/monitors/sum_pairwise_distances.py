"""
Monitor: sum_pairwise_distances_variance

Checks whether r12 + r13 + r23 is approximately conserved along a trajectory.

Discovery provenance: Phase 2, 2026-03-13. First dual-filter pass in the 3-body
conservation law search. Oracle margin +4.50 on figure-8; frac_var = 5.54e-04.
Mechanism: Z3 choreographic symmetry cancels 1ω₀ and 2ω₀ Fourier harmonics.

Threshold tuning:
  < 1e-3  → strong conservation (figure-8 class)
  1e-3 to 1e-2 → weak near-conservation (quadratic Z3 forms)
  > 1e-2  → not conserved (random / hierarchical ICs baseline ~ 0.3–0.9)

Usage in problem.yaml:
  monitors:
    - margin_sign
    - sum_pairwise_distances_variance
"""

import numpy as np
from typing import Dict


def compute_sum_pairwise_distances_variance(positions: np.ndarray) -> Dict:
    """
    Check whether r12 + r13 + r23 is approximately conserved.

    Args:
        positions: shape (T, 3, D) — T timesteps, 3 bodies, D spatial dims

    Returns dict with:
        sum_pairwise_distances: mean value of r12+r13+r23 over the trajectory
        frac_var: fractional variation σ / |mean| (lower = more conserved)
        is_conserved: True if frac_var < 1e-3
        better_than_energy: True if frac_var < 5e-4 (comparable to energy integrator error)
    """
    T = positions.shape[0]
    sum_dist = np.zeros(T)

    for t in range(T):
        p1, p2, p3 = positions[t]
        r12 = np.linalg.norm(p1 - p2)
        r13 = np.linalg.norm(p1 - p3)
        r23 = np.linalg.norm(p2 - p3)
        sum_dist[t] = r12 + r13 + r23

    mean = np.mean(sum_dist)
    frac_var = np.std(sum_dist) / mean if mean > 0 else 0.0

    return {
        "sum_pairwise_distances": float(mean),
        "frac_var": float(frac_var),
        "is_conserved": bool(frac_var < 1e-3),
        "better_than_energy": bool(frac_var < 5e-4),
    }


# Vectorised version — faster for long integrations (positions already as array)
def compute_vectorised(positions: np.ndarray) -> Dict:
    """Same as above but without the Python loop. Preferred for T > 1000."""
    p1 = positions[:, 0, :]
    p2 = positions[:, 1, :]
    p3 = positions[:, 2, :]
    r12 = np.linalg.norm(p1 - p2, axis=1)
    r13 = np.linalg.norm(p1 - p3, axis=1)
    r23 = np.linalg.norm(p2 - p3, axis=1)
    sum_dist = r12 + r13 + r23
    mean = np.mean(sum_dist)
    frac_var = np.std(sum_dist) / mean if mean > 0 else 0.0
    return {
        "sum_pairwise_distances": float(mean),
        "frac_var": float(frac_var),
        "is_conserved": bool(frac_var < 1e-3),
        "better_than_energy": bool(frac_var < 5e-4),
    }


if __name__ == "__main__":
    # Quick self-test on figure-8 ICs
    import sys
    sys.path.insert(0, "..")
    from conservation_checker import integrate_3body, ic_figure8, ic_general_random

    print("sum_pairwise_distances monitor — self-test")
    print()

    for name, ic_fn in [("figure-8", ic_figure8), ("random(42)", lambda: ic_general_random(42))]:
        masses, pos0, vel0 = ic_fn()
        t, state = integrate_3body(masses, pos0, vel0, t_end=100, n_points=5000)
        N = 3
        positions = state[:, :3*N].reshape(-1, N, 3)
        result = compute_vectorised(positions)
        print(f"{name:15s}  frac_var={result['frac_var']:.3e}  "
              f"conserved={result['is_conserved']}  "
              f"better_than_energy={result['better_than_energy']}")
