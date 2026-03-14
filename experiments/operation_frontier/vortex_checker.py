#!/usr/bin/env python3
"""
Numerical conservation checker for the 2D point-vortex problem.

The 2D point-vortex model is the simplest reduced fluid system with known
exact conservation laws and potential approximate ones. It's the natural
next domain after the 3-body gravitational problem.

Physical setup:
  N point vortices at positions (xᵢ, yᵢ) with circulations Γᵢ.
  Equations of motion (Kirchhoff 1876):
    dxᵢ/dt = -1/(2π) Σⱼ≠ᵢ Γⱼ (yᵢ-yⱼ)/rᵢⱼ²
    dyᵢ/dt =  1/(2π) Σⱼ≠ᵢ Γⱼ (xᵢ-xⱼ)/rᵢⱼ²

Exact conserved quantities (known):
  H  = -1/(4π) Σᵢ<ⱼ Γᵢ Γⱼ ln(rᵢⱼ²)            (Hamiltonian)
  Lz = Σᵢ Γᵢ (xᵢ² + yᵢ²)                       (angular momentum)
  Px = Σᵢ Γᵢ yᵢ,  Py = -Σᵢ Γᵢ xᵢ              (linear impulse)
  Γ_total = Σᵢ Γᵢ                                (total circulation, trivial)

Open questions / hunt targets:
  - For near-equal-circulation vortex pair undergoing slow merger: is there
    a near-invariant analogous to the figure-8 choreographic invariants?
  - Does r12 = const for equal-strength vortex pair? (YES — trivially, so check
    higher-order generalizations)
  - For 3-vortex system with one near-zero circulation (restricted vortex problem):
    is there a Jacobi-like approximate integral?

Usage:
    # Check known conserved quantities:
    python vortex_checker.py --all

    # Check a custom expression:
    python vortex_checker.py --expr "s['r12']"

    # Check specific IC type:
    python vortex_checker.py --check H --ic equal_pair
    python vortex_checker.py --check H --ic unequal_pair
    python vortex_checker.py --check H --ic three_vortex
"""

import argparse
import numpy as np
from scipy.integrate import solve_ivp


# --------------------------------------------------------------------------
# Point-vortex integrator
# --------------------------------------------------------------------------

def vortex_rhs(t, state, circulations):
    """Kirchhoff equations for N point vortices."""
    N = len(circulations)
    pos = state.reshape(N, 2)
    G   = circulations
    dpos = np.zeros_like(pos)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            r2 = dx**2 + dy**2 + 1e-20
            dpos[i, 0] += -G[j] * dy / (2 * np.pi * r2)
            dpos[i, 1] +=  G[j] * dx / (2 * np.pi * r2)
    return dpos.ravel()


def integrate_vortex(circulations, pos0, t_end=50.0, n_points=2000):
    """Integrate point-vortex system. Returns (t, state) arrays."""
    state0 = pos0.ravel()
    t_span = (0.0, t_end)
    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(vortex_rhs, t_span, state0, args=(circulations,),
                    method="RK45", t_eval=t_eval, rtol=1e-10, atol=1e-12,
                    dense_output=False)
    return sol.t, sol.y.T   # (n_points,), (n_points, 2*N)


# --------------------------------------------------------------------------
# Initial conditions
# --------------------------------------------------------------------------

def ic_equal_pair(sep=2.0):
    """Two equal-strength co-rotating vortices. r12 = const (exact)."""
    G = np.array([1.0, 1.0])
    pos0 = np.array([[-sep/2, 0.0], [sep/2, 0.0]])
    return G, pos0


def ic_unequal_pair(sep=2.0, ratio=0.8):
    """Two co-rotating vortices with slightly unequal circulations.
    r12 NOT constant — vortex centroid still fixed, vortices spiral."""
    G = np.array([1.0, ratio])
    pos0 = np.array([[-sep/2, 0.0], [sep/2, 0.0]])
    return G, pos0


def ic_opposite_pair(sep=2.0):
    """Two equal and opposite vortices (vortex dipole). Translates at constant velocity."""
    G = np.array([1.0, -1.0])
    pos0 = np.array([[-sep/2, 0.0], [sep/2, 0.0]])
    return G, pos0


def ic_three_vortex_symmetric():
    """Three equal-circulation vortices at vertices of equilateral triangle.
    This is the point-vortex analog of the equilateral Lagrange configuration."""
    G = np.array([1.0, 1.0, 1.0])
    r = 1.5
    pos0 = np.array([
        [r * np.cos(0),             r * np.sin(0)],
        [r * np.cos(2*np.pi/3),     r * np.sin(2*np.pi/3)],
        [r * np.cos(4*np.pi/3),     r * np.sin(4*np.pi/3)],
    ])
    return G, pos0


def ic_three_vortex_random(seed=42):
    """Random three-vortex configuration. Near-zero total circulation."""
    rng = np.random.default_rng(seed)
    G = rng.uniform(0.5, 1.5, 3)
    G -= G.mean()   # zero total circulation
    pos0 = rng.standard_normal((3, 2)) * 1.5
    return G, pos0


def ic_restricted_three_vortex():
    """Two strong vortices + one weak test vortex (restricted problem)."""
    G = np.array([1.0, 1.0, 0.01])
    pos0 = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    return G, pos0


# --------------------------------------------------------------------------
# Parse state
# --------------------------------------------------------------------------

def parse_state(t, state, circulations):
    """Extract named quantities from vortex state."""
    N = len(circulations)
    pos = state.reshape(-1, N, 2)   # (n_t, N, 2)
    G   = circulations

    x = pos[:, :, 0]   # (n_t, N)
    y = pos[:, :, 1]

    # Pairwise distances
    d = {}
    for i in range(N):
        for j in range(i+1, N):
            dx = x[:, i] - x[:, j]
            dy = y[:, i] - y[:, j]
            d[f"r{i+1}{j+1}"] = np.sqrt(dx**2 + dy**2)

    # Hamiltonian  H = -1/(4π) Σᵢ<ⱼ Γᵢ Γⱼ ln(rᵢⱼ²)
    H = np.zeros(len(t))
    for i in range(N):
        for j in range(i+1, N):
            rij = d[f"r{i+1}{j+1}"]
            H += -G[i] * G[j] / (4 * np.pi) * np.log(rij**2 + 1e-30)

    # Angular momentum  Lz = Σᵢ Γᵢ (xᵢ² + yᵢ²)
    Lz = np.sum(G[None, :] * (x**2 + y**2), axis=1)

    # Linear impulse  Px = Σᵢ Γᵢ yᵢ,  Py = -Σᵢ Γᵢ xᵢ
    Px = np.sum(G[None, :] * y, axis=1)
    Py = np.sum(G[None, :] * (-x), axis=1)

    # Total circulation (trivial constant)
    Gamma_total = float(np.sum(G))

    # Vorticity-weighted centroid
    G_total = np.sum(G)
    if abs(G_total) > 1e-10:
        Xcm = np.sum(G[None, :] * x, axis=1) / G_total
        Ycm = np.sum(G[None, :] * y, axis=1) / G_total
    else:
        Xcm = np.zeros(len(t))
        Ycm = np.zeros(len(t))

    s = dict(
        t=t, pos=pos, x=x, y=y, G=G,
        H=H, Lz=Lz, Px=Px, Py=Py,
        Gamma_total=Gamma_total,
        Xcm=Xcm, Ycm=Ycm,
        **d,
    )
    return s


def frac_var(arr):
    """Fractional variation σ/|mean|."""
    mean = np.mean(arr)
    if abs(mean) < 1e-15:
        return float(np.std(arr))
    return float(np.std(arr) / abs(mean))


# --------------------------------------------------------------------------
# Built-in candidates
# --------------------------------------------------------------------------

BUILTIN_CANDIDATES = {
    "H": {
        "fn": lambda s: s["H"],
        "desc": "Hamiltonian H = -1/(4π) Σ Γᵢ Γⱼ ln(rᵢⱼ²)",
        "expected": "conserved (exact)",
    },
    "Lz": {
        "fn": lambda s: s["Lz"],
        "desc": "Angular momentum Lz = Σ Γᵢ (xᵢ²+yᵢ²)",
        "expected": "conserved (exact)",
    },
    "Px": {
        "fn": lambda s: s["Px"],
        "desc": "Linear impulse Px = Σ Γᵢ yᵢ",
        "expected": "conserved (exact)",
    },
    "Py": {
        "fn": lambda s: s["Py"],
        "desc": "Linear impulse Py = -Σ Γᵢ xᵢ",
        "expected": "conserved (exact)",
    },
    "r12": {
        "fn": lambda s: s["r12"],
        "desc": "Separation r12 (conserved exactly for equal-circulation pair)",
        "expected": "PASS equal_pair, FAIL unequal_pair",
    },
    "Xcm": {
        "fn": lambda s: s["Xcm"],
        "desc": "Vorticity-weighted centroid X",
        "expected": "conserved (exact, Γ_total ≠ 0)",
    },
    "r12_sq": {
        "fn": lambda s: s["r12"]**2,
        "desc": "r12² (same as r12 for equal pair)",
        "expected": "PASS equal_pair",
    },
}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def run_all(ic_name, G, pos0, t_end=50.0, threshold=1e-3):
    t, state = integrate_vortex(G, pos0, t_end=t_end)
    s = parse_state(t, state, G)
    N = len(G)
    print(f"\nIC: {ic_name}  N={N}  Γ={G}  t_end={t_end}")
    print(f"{'Candidate':30s}  {'frac_var':>12s}  {'verdict':>8s}")
    print("-" * 58)
    for name, cand in BUILTIN_CANDIDATES.items():
        try:
            vals = cand["fn"](s)
            fv = frac_var(vals)
            v = "PASS" if fv < threshold else "fail"
            print(f"{name:30s}  {fv:12.3e}  {v:>8s}  # {cand['expected']}")
        except KeyError:
            print(f"{name:30s}  {'N/A':>12s}  {'skip':>8s}")


IC_MAP = {
    "equal_pair":           ic_equal_pair,
    "unequal_pair":         ic_unequal_pair,
    "opposite_pair":        ic_opposite_pair,
    "three_symmetric":      ic_three_vortex_symmetric,
    "three_random":         ic_three_vortex_random,
    "restricted":           ic_restricted_three_vortex,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Point-vortex conservation checker")
    parser.add_argument("--all",    action="store_true", help="Run all ICs and candidates")
    parser.add_argument("--ic",     default="equal_pair", choices=list(IC_MAP.keys()))
    parser.add_argument("--check",  default=None, help="Check a built-in candidate by name")
    parser.add_argument("--expr",   default=None, help="Python expression in s['...'] namespace")
    parser.add_argument("--t-end",  type=float, default=50.0)
    parser.add_argument("--threshold", type=float, default=1e-3)
    args = parser.parse_args()

    if args.all:
        for ic_name, ic_fn in IC_MAP.items():
            G, pos0 = ic_fn()
            run_all(ic_name, G, pos0, t_end=args.t_end, threshold=args.threshold)
    else:
        G, pos0 = IC_MAP[args.ic]()
        t, state = integrate_vortex(G, pos0, t_end=args.t_end)
        s = parse_state(t, state, G)

        if args.check:
            cand = BUILTIN_CANDIDATES[args.check]
            vals = cand["fn"](s)
            fv = frac_var(vals)
            v = "PASS" if fv < args.threshold else "FAIL"
            print(f"{args.ic} / {args.check}: frac_var={fv:.3e}  {v}")

        elif args.expr:
            vals = eval(args.expr, {"s": s, "np": np})
            fv = frac_var(np.asarray(vals))
            v = "PASS" if fv < args.threshold else "FAIL"
            print(f"{args.ic} / expr: frac_var={fv:.3e}  {v}")
            print(f"  expr: {args.expr}")

        else:
            run_all(args.ic, G, pos0, t_end=args.t_end, threshold=args.threshold)
