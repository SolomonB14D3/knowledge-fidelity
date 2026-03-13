#!/usr/bin/env python3
"""
Numerical conservation checker for Operation Frontier.

Given a candidate expression (as a Python lambda or string), integrates
3-body gravitational trajectories and checks whether the expression is
approximately conserved (d/dt ≈ 0) along them.

This is the FORMAL CHECKER in the dual-filter pipeline:
  1. Oracle filter (oracle_wrapper.py) — fast, cheap, language-model-based
  2. Formal filter (this script)       — slower, rigorous, numerically verified

A candidate passes the formal check if its fractional variation
  sigma(f) / |mean(f)|  <  threshold
along >= 3 independent trajectories.

Pipeline position:
  Oracle PASS + Checker PASS  →  Strong candidate (known-consistent + numerically conserved)
  Oracle FAIL + Checker PASS  →  MOST INTERESTING (numerically real but model hasn't seen it)
  Oracle PASS + Checker FAIL  →  Plausible but wrong (model thinks it's right, it's not)
  Oracle FAIL + Checker FAIL  →  Dead end

Usage:
    # Check that total energy is conserved (sanity test):
    python conservation_checker.py --check energy

    # Check Jacobi constant in CRTBP:
    python conservation_checker.py --check jacobi

    # Check a custom candidate expression:
    python conservation_checker.py --expr "m1*v1**2 + m2*v2**2 + m3*v3**2"

    # Run all built-in candidates and show which pass:
    python conservation_checker.py --all
"""

import argparse
import numpy as np
from scipy.integrate import solve_ivp


# --------------------------------------------------------------------------
# N-body integrator (3 bodies, Newtonian gravity)
# --------------------------------------------------------------------------

G = 1.0   # Gravitational constant (natural units)


def nbody_rhs(t, state, masses):
    """Right-hand side of Newton's equations for N gravitating bodies.

    state: [x1,y1,z1, x2,y2,z2, ..., vx1,vy1,vz1, vx2,vy2,vz2, ...]
    Returns: d(state)/dt
    """
    N = len(masses)
    pos = state[:3*N].reshape(N, 3)
    vel = state[3*N:].reshape(N, 3)

    acc = np.zeros((N, 3))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dr = pos[j] - pos[i]
            r3 = np.dot(dr, dr)**1.5 + 1e-20
            acc[i] += G * masses[j] * dr / r3

    return np.concatenate([vel.flatten(), acc.flatten()])


def integrate_3body(masses, pos0, vel0, t_end=50.0, n_points=2000):
    """Integrate a 3-body system and return trajectory array.

    Returns:
        t:     (n_points,) time array
        state: (n_points, 18) state array [x,y,z x3, vx,vy,vz x3]
    """
    state0 = np.concatenate([pos0.flatten(), vel0.flatten()])
    t_eval = np.linspace(0, t_end, n_points)
    sol = solve_ivp(
        nbody_rhs, [0, t_end], state0, args=(masses,),
        t_eval=t_eval, method='RK45',
        rtol=1e-10, atol=1e-12,
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return sol.t, sol.y.T   # (n_points, 18)


# --------------------------------------------------------------------------
# Initial condition generators
# --------------------------------------------------------------------------

def ic_general_random(seed=42):
    """Random 3-body initial conditions with zero total momentum."""
    rng = np.random.default_rng(seed)
    masses = np.array([1.0, 1.0, 1.0])
    pos0 = rng.standard_normal((3, 3)) * 2.0
    vel0 = rng.standard_normal((3, 3)) * 0.5
    # Zero total momentum
    vel0 -= (masses[:, None] * vel0).sum(0) / masses.sum()
    # Zero center of mass
    pos0 -= (masses[:, None] * pos0).sum(0) / masses.sum()
    return masses, pos0, vel0


def ic_figure8():
    """Figure-8 choreographic 3-body orbit (Chenciner & Montgomery 2000).
    Equal masses, planar, all bodies trace same figure-8 curve.
    Initial conditions from Simó (2002) normalisation."""
    masses = np.array([1.0, 1.0, 1.0])
    # From Simó 2002 table: positions and velocities in natural units
    x1, y1 = 0.97000436, -0.24308753
    vx3, vy3 = -0.93240737 / 2, -0.86473146 / 2
    pos0 = np.array([
        [ x1,  y1, 0.0],
        [-x1, -y1, 0.0],
        [ 0.0, 0.0, 0.0],
    ])
    vel0 = np.array([
        [-vx3, -vy3, 0.0],   # actually vx1 = -vx3 by symmetry
        [-vx3, -vy3, 0.0],
        [ 2*vx3, 2*vy3, 0.0],
    ])
    # Apply correct figure-8 velocities
    # Standard Chenciner-Montgomery ICs
    vx1 = 0.93240737 / 2
    vy1 = 0.86473146 / 2
    vel0 = np.array([
        [ vx1,  vy1, 0.0],
        [ vx1,  vy1, 0.0],
        [-2*vx1, -2*vy1, 0.0],
    ])
    return masses, pos0, vel0


def ic_hierarchical(inner_sep=1.0, outer_sep=10.0, seed=0):
    """Hierarchical triple: tight inner binary + distant third body."""
    masses = np.array([1.0, 1.0, 0.1])
    # Inner binary in circular orbit
    r_in = inner_sep
    v_in = np.sqrt(G * (masses[0] + masses[1]) / (2 * r_in))
    # Outer body in circular orbit
    r_out = outer_sep
    v_out = np.sqrt(G * (masses[0] + masses[1] + masses[2]) / r_out)
    pos0 = np.array([
        [ r_in,  0.0, 0.0],
        [-r_in,  0.0, 0.0],
        [ 0.0, r_out, 0.0],
    ])
    vel0 = np.array([
        [0.0,  v_in, 0.0],
        [0.0, -v_in, 0.0],
        [-v_out, 0.0, 0.0],
    ])
    return masses, pos0, vel0


# --------------------------------------------------------------------------
# Candidate expressions
# --------------------------------------------------------------------------

def parse_state(state, masses):
    """Extract named quantities from state vector."""
    N = 3
    pos = state[:, :3*N].reshape(-1, N, 3)
    vel = state[:, 3*N:].reshape(-1, N, 3)
    m = masses

    # Positions
    r = np.linalg.norm(pos, axis=2)  # (n, 3)
    # Pairwise distances
    r12 = np.linalg.norm(pos[:,0] - pos[:,1], axis=1)
    r13 = np.linalg.norm(pos[:,0] - pos[:,2], axis=1)
    r23 = np.linalg.norm(pos[:,1] - pos[:,2], axis=1)
    # Speeds
    v = np.linalg.norm(vel, axis=2)  # (n, 3)
    # Kinetic energy
    KE = 0.5 * np.sum(m[None,:] * v**2, axis=1)
    # Potential energy
    PE = -(G*m[0]*m[1]/r12 + G*m[0]*m[2]/r13 + G*m[1]*m[2]/r23)
    # Total energy
    E = KE + PE
    # Linear momentum components
    P = np.sum(m[None,:,None] * vel, axis=1)  # (n, 3)
    Px, Py, Pz = P[:,0], P[:,1], P[:,2]
    # Angular momentum
    L_vec = np.sum(m[None,:,None] * np.cross(pos, vel), axis=1)  # (n, 3)
    Lz = L_vec[:,2]
    L_mag = np.linalg.norm(L_vec, axis=1)
    # Center of mass
    R_cm = np.sum(m[None,:,None] * pos, axis=1) / m.sum()

    return dict(
        pos=pos, vel=vel, m=m,
        r=r, r12=r12, r13=r13, r23=r23,
        v=v, KE=KE, PE=PE, E=E,
        Px=Px, Py=Py, Pz=Pz,
        Lz=Lz, L_mag=L_mag, L_vec=L_vec,
        R_cm=R_cm,
        # Convenient shorthands
        m1=m[0], m2=m[1], m3=m[2],
        x1=pos[:,0,0], y1=pos[:,0,1], z1=pos[:,0,2],
        x2=pos[:,1,0], y2=pos[:,1,1], z2=pos[:,1,2],
        x3=pos[:,2,0], y3=pos[:,2,1], z3=pos[:,2,2],
        vx1=vel[:,0,0], vy1=vel[:,0,1], vz1=vel[:,0,2],
        vx2=vel[:,1,0], vy2=vel[:,1,1], vz2=vel[:,1,2],
        vx3=vel[:,2,0], vy3=vel[:,2,1], vz3=vel[:,2,2],
        v1=v[:,0], v2=v[:,1], v3=v[:,2],
    )


# Built-in candidates
BUILTIN_CANDIDATES = {
    "energy": {
        "fn": lambda s: s["E"],
        "description": "Total energy E = KE + PE",
        "expected": "conserved",
    },
    "px": {
        "fn": lambda s: s["Px"],
        "description": "Total x-momentum",
        "expected": "conserved",
    },
    "lz": {
        "fn": lambda s: s["Lz"],
        "description": "Total z-angular momentum",
        "expected": "conserved",
    },
    "l_mag": {
        "fn": lambda s: s["L_mag"],
        "description": "|L| total angular momentum magnitude",
        "expected": "conserved",
    },
    "ke_only": {
        "fn": lambda s: s["KE"],
        "description": "Kinetic energy alone (NOT conserved)",
        "expected": "NOT conserved",
    },
    "r12_r13": {
        "fn": lambda s: s["r12"] + s["r13"],
        "description": "r12 + r13 (NOT a conserved quantity)",
        "expected": "NOT conserved",
    },
    "virial": {
        "fn": lambda s: 2*s["KE"] + s["PE"],
        "description": "Virial quantity 2T+V (NOT instantaneously conserved, time-average 0)",
        "expected": "NOT conserved (oscillates around 0)",
    },
    "lagrange_r12_over_r23": {
        "fn": lambda s: s["r12"] / s["r23"],
        "description": "r12/r23 ratio (conserved in equilateral Lagrange config, not general)",
        "expected": "varies",
    },
    # Candidate: is (r12 * r13 * r23) conserved? (product of all pairwise distances)
    "r_product": {
        "fn": lambda s: s["r12"] * s["r13"] * s["r23"],
        "description": "r12*r13*r23 (product of pairwise distances) — candidate",
        "expected": "unknown",
    },
    # Candidate: moment of inertia I = sum m_i r_i^2
    "inertia": {
        "fn": lambda s: np.sum(s["m"][None,:] * np.sum(s["pos"]**2, axis=2), axis=1),
        "description": "Moment of inertia I = sum m_i r_i^2 — NOT conserved, obeys Lagrange-Jacobi",
        "expected": "NOT conserved (Lagrange-Jacobi: I'' = 4E - 2V)",
    },
    # Candidate: shape invariant in equal-mass equilateral config
    "shape_equilateral": {
        "fn": lambda s: (s["r12"] - s["r13"])**2 + (s["r13"] - s["r23"])**2 + (s["r12"] - s["r23"])**2,
        "description": "Sum of squared pairwise distance differences — near-zero iff equilateral",
        "expected": "NOT conserved in general",
    },
}


# --------------------------------------------------------------------------
# Conservation test
# --------------------------------------------------------------------------

def test_conservation(fn, t, state, masses, threshold=1e-3, label="candidate"):
    """Test whether fn(state) is approximately conserved along a trajectory.

    Returns dict with:
      conserved: bool
      frac_variation: sigma/|mean|
      initial_value: fn(state[0])
      final_value: fn(state[-1])
      drift: (final - initial) / |initial|
    """
    s = parse_state(state, masses)
    values = fn(s)

    mean_val = np.mean(values)
    std_val  = np.std(values)
    initial  = values[0]
    final    = values[-1]

    if abs(mean_val) < 1e-15:
        frac_var = std_val
    else:
        frac_var = std_val / abs(mean_val)

    drift = (final - initial) / (abs(initial) + 1e-15)

    conserved = frac_var < threshold

    return {
        "conserved": conserved,
        "frac_variation": frac_var,
        "threshold": threshold,
        "initial": float(initial),
        "final": float(final),
        "mean": float(mean_val),
        "drift": float(drift),
        "label": label,
    }


def run_check(name_or_fn, description="", threshold=1e-3, t_end=50.0):
    """Run a candidate against multiple initial conditions."""
    if callable(name_or_fn):
        fn = name_or_fn
        desc = description
    else:
        info = BUILTIN_CANDIDATES[name_or_fn]
        fn = info["fn"]
        desc = info["description"]
        expected = info.get("expected", "unknown")
        print(f"\n{'─'*65}")
        print(f"  Candidate: {name_or_fn}")
        print(f"  {desc}")
        print(f"  Expected: {expected}")
        print(f"{'─'*65}")

    ics = [
        ("random_1",       *ic_general_random(seed=1)),
        ("random_2",       *ic_general_random(seed=7)),
        ("random_3",       *ic_general_random(seed=42)),
        ("figure8",        *ic_figure8()),
        ("hierarchical",   *ic_hierarchical()),
    ]

    results = []
    for ic_name, masses, pos0, vel0 in ics:
        try:
            t, state = integrate_3body(masses, pos0, vel0, t_end=t_end)
            r = test_conservation(fn, t, state, masses, threshold=threshold)
            r["ic"] = ic_name
            results.append(r)
            mark = "✓" if r["conserved"] else "✗"
            print(f"  [{ic_name:14s}] {mark}  frac_var={r['frac_variation']:.2e}  "
                  f"drift={r['drift']:+.2e}  init={r['initial']:.4f}")
        except Exception as e:
            print(f"  [{ic_name:14s}] ERROR: {e}")
            results.append({"ic": ic_name, "error": str(e), "conserved": False})

    n_pass = sum(r.get("conserved", False) for r in results)
    n_total = len(results)
    overall = n_pass >= 3
    print(f"\n  → {n_pass}/{n_total} trajectories conserved.  "
          f"{'PASS ✓' if overall else 'FAIL ✗'}")
    return results, overall


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=str, default=None,
                        help=f"Name of built-in candidate: {list(BUILTIN_CANDIDATES)}")
    parser.add_argument("--all", action="store_true",
                        help="Run all built-in candidates")
    parser.add_argument("--expr", type=str, default=None,
                        help="Python expression using state variables (m1,m2,m3,r12,r13,r23,v1,v2,v3,x1..z3,KE,PE,E,...)")
    parser.add_argument("--threshold", type=float, default=1e-3,
                        help="Fractional variation threshold for conservation (default 1e-3)")
    parser.add_argument("--t-end", type=float, default=50.0,
                        help="Integration end time (default 50.0)")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  Conservation Checker — Operation Frontier")
    print("  Formal verifier: d(candidate)/dt ≈ 0 along integrated trajectories")
    print("="*65)

    if args.all:
        summary = []
        for name in BUILTIN_CANDIDATES:
            results, passed = run_check(name, threshold=args.threshold, t_end=args.t_end)
            summary.append((name, passed))
        print(f"\n{'='*65}")
        print(f"  Summary:")
        for name, passed in summary:
            print(f"    {'PASS ✓' if passed else 'FAIL ✗'}  {name}")
        return

    if args.check:
        if args.check not in BUILTIN_CANDIDATES:
            print(f"Unknown candidate '{args.check}'. Available: {list(BUILTIN_CANDIDATES)}")
            return
        run_check(args.check, threshold=args.threshold, t_end=args.t_end)
        return

    if args.expr:
        print(f"\nEvaluating expression: {args.expr}")
        def expr_fn(s):
            return eval(args.expr, {"__builtins__": {}}, {**s, "np": np})
        results, passed = run_check(expr_fn, description=args.expr,
                                    threshold=args.threshold, t_end=args.t_end)
        return

    # Default: run the sanity tests (energy + known non-conserved)
    print("\nRunning sanity checks (energy should pass, KE alone should fail)...")
    run_check("energy", threshold=args.threshold, t_end=args.t_end)
    run_check("lz", threshold=args.threshold, t_end=args.t_end)
    run_check("ke_only", threshold=args.threshold, t_end=args.t_end)


if __name__ == "__main__":
    main()
