#!/usr/bin/env python3
"""Build master SQLite database from all JSON result files.

Consolidates 56+ JSON result files into a single queryable SQLite database.
Tables are designed for common analytical queries (e.g., "all factual rho
scores across experiments", "hybrid sweep configs ranked by mean improvement").

Usage:
    python scripts/build_master_db.py                    # default: results/master.db
    python scripts/build_master_db.py -o my_results.db   # custom output
    python scripts/build_master_db.py --dry-run           # show what would be ingested

Tables:
    alignment_runs     — SFT sweep/ablation runs (model, condition, seed, rho scores)
    hybrid_sweep       — Hybrid control configs (SVD × SAE × rho-SFT grid)
    steering_heatmap   — Layer-wise steering experiments (per-layer, per-behavior)
    attack_defense     — SAE steering sweep results (per-scale)
    freeze_sweep       — Compression ratio × freeze ratio grid
    leaderboard        — Merge method audit scores
    fidelity_bench     — Fidelity-Bench category scores
    cf90_multiseed     — CF90 compression multi-seed results
    metadata           — Source file tracking (file, table, ingested_at)
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_DB = RESULTS_DIR / "master.db"

BEHAVIORS = [
    "factual", "toxicity", "bias", "sycophancy",
    "reasoning", "refusal", "deception", "overrefusal",
]


def create_tables(conn: sqlite3.Connection):
    """Create all tables."""
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            target_table TEXT NOT NULL,
            rows_inserted INTEGER DEFAULT 0,
            ingested_at TEXT NOT NULL
        )
    """)

    # ── alignment_runs: one row per (model, condition, seed, rho_weight) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS alignment_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT,
            model TEXT,
            backend TEXT,
            experiment TEXT,
            condition TEXT,
            seed INTEGER,
            rho_weight REAL,
            sft_size INTEGER,
            lora_rank INTEGER,
            epochs INTEGER,
            lr REAL,
            margin REAL,
            elapsed_seconds REAL,
            train_ce_loss REAL,
            train_rho_loss REAL,
            train_steps INTEGER,
            -- baseline scores
            bl_factual REAL, bl_toxicity REAL, bl_bias REAL,
            bl_sycophancy REAL, bl_reasoning REAL, bl_refusal REAL,
            bl_deception REAL, bl_overrefusal REAL,
            -- post-intervention scores
            factual REAL, toxicity REAL, bias REAL,
            sycophancy REAL, reasoning REAL, refusal REAL,
            deception REAL, overrefusal REAL,
            -- deltas
            d_factual REAL, d_toxicity REAL, d_bias REAL,
            d_sycophancy REAL, d_reasoning REAL, d_refusal REAL,
            d_deception REAL, d_overrefusal REAL
        )
    """)

    # ── hybrid_sweep: one row per config ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS hybrid_sweep (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT,
            model_name TEXT,
            config_tag TEXT,
            timestamp TEXT,
            compress_ratio REAL,
            freeze_fraction REAL,
            sae_layer INTEGER,
            rho_weight REAL,
            seed INTEGER,
            total_elapsed_sec REAL,
            -- audit_before
            before_factual REAL, before_toxicity REAL, before_bias REAL,
            before_sycophancy REAL, before_reasoning REAL, before_refusal REAL,
            before_deception REAL, before_overrefusal REAL,
            -- audit_after
            after_factual REAL, after_toxicity REAL, after_bias REAL,
            after_sycophancy REAL, after_reasoning REAL, after_refusal REAL,
            after_deception REAL, after_overrefusal REAL,
            -- deltas
            d_factual REAL, d_toxicity REAL, d_bias REAL,
            d_sycophancy REAL, d_reasoning REAL, d_refusal REAL,
            d_deception REAL, d_overrefusal REAL,
            mean_before REAL,
            mean_after REAL,
            mean_delta REAL
        )
    """)

    # ── steering_heatmap: one row per (model, layer, behavior) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS steering_heatmap (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT,
            model_id TEXT,
            experiment TEXT,
            layer INTEGER,
            depth_pct REAL,
            alpha REAL,
            vector_norm REAL,
            elapsed_seconds REAL,
            behavior TEXT,
            rho_baseline REAL,
            rho_steered REAL,
            rho_delta REAL
        )
    """)

    # ── attack_defense: one row per (behavior, scale) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS attack_defense (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT,
            model TEXT,
            layer INTEGER,
            behavior TEXT,
            status TEXT,
            n_features INTEGER,
            scale REAL,
            rho REAL,
            delta REAL,
            n_regressed INTEGER,
            mean_collateral REAL
        )
    """)

    # ── freeze_sweep: one row per (model, compress_ratio, freeze_ratio, behavior) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS freeze_sweep (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT,
            model TEXT,
            model_id TEXT,
            compress_ratio REAL,
            freeze_ratio REAL,
            behavior TEXT,
            rho_baseline REAL,
            rho_compressed REAL,
            rho_delta REAL,
            retention REAL,
            positive_count INTEGER,
            total INTEGER
        )
    """)

    # ── leaderboard: one row per (model, behavior) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS leaderboard (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT,
            model_key TEXT,
            model_id TEXT,
            model_type TEXT,
            description TEXT,
            device TEXT,
            timestamp TEXT,
            behavior TEXT,
            rho REAL,
            retention REAL,
            positive_count INTEGER,
            total INTEGER
        )
    """)

    # ── fidelity_bench: one row per (model, category) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS fidelity_bench (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT,
            model TEXT,
            category TEXT,
            n_probes INTEGER,
            rho REAL,
            rho_p REAL,
            mean_delta REAL,
            accuracy REAL,
            elapsed_seconds REAL
        )
    """)

    # ── cf90_multiseed: one row per (model, seed) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS cf90_multiseed (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT,
            model TEXT,
            ratio REAL,
            seed INTEGER,
            retention REAL,
            rho_before REAL,
            rho_after REAL,
            rho_drop REAL,
            n_compressed INTEGER,
            elapsed_seconds REAL
        )
    """)

    conn.commit()


def log_metadata(conn, source_file: str, table: str, n_rows: int):
    conn.execute(
        "INSERT INTO metadata (source_file, target_table, rows_inserted, ingested_at) VALUES (?, ?, ?, ?)",
        (source_file, table, n_rows, datetime.now().isoformat()),
    )


def load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


def safe_get(d: dict, key: str, default=None):
    """Safely get a nested value."""
    return d.get(key, default) if isinstance(d, dict) else default


# ── Ingestors ──────────────────────────────────────────────────────────

def ingest_alignment(conn: sqlite3.Connection, dry_run: bool = False):
    """Ingest results/alignment/*.json into alignment_runs."""
    alignment_dir = RESULTS_DIR / "alignment"
    if not alignment_dir.exists():
        return

    total = 0
    for fpath in sorted(alignment_dir.glob("*.json")):
        data = load_json(fpath)
        rel = str(fpath.relative_to(RESULTS_DIR))

        model = data.get("model", "")
        backend = data.get("backend", "")
        experiment = data.get("experiment", fpath.stem)
        config = data.get("config", {})

        # Get baseline scores
        bl = data.get("baseline_quick", data.get("baseline", {}))

        # Handle two schemas: "runs" list or "merged_deltas" dict
        runs = data.get("runs", [])
        if not runs and "merged_deltas" in data:
            md = data["merged_deltas"]
            for cond, beh_dict in md.items():
                n_seeds = len(next(iter(beh_dict.values())))
                for i in range(n_seeds):
                    seeds = data.get("seeds", list(range(n_seeds)))
                    seed = seeds[i] if i < len(seeds) else i
                    row = {
                        "source_file": rel,
                        "model": model,
                        "backend": backend,
                        "experiment": experiment,
                        "condition": cond,
                        "seed": seed,
                        "rho_weight": config.get("rho_weight"),
                        "sft_size": config.get("sft_size"),
                        "lora_rank": config.get("lora_rank"),
                        "epochs": config.get("epochs"),
                        "lr": config.get("lr"),
                        "margin": config.get("margin"),
                        "elapsed_seconds": None,
                        "train_ce_loss": None,
                        "train_rho_loss": None,
                        "train_steps": None,
                    }
                    for b in BEHAVIORS:
                        row[f"bl_{b}"] = bl.get(b)
                        delta = beh_dict.get(b, [0.0] * n_seeds)
                        d_val = delta[i] if i < len(delta) else None
                        row[f"d_{b}"] = d_val
                        bl_val = bl.get(b, 0.0)
                        row[b] = (bl_val + d_val) if d_val is not None and bl_val is not None else None

                    if not dry_run:
                        cols = list(row.keys())
                        placeholders = ", ".join(["?"] * len(cols))
                        conn.execute(
                            f"INSERT INTO alignment_runs ({', '.join(cols)}) VALUES ({placeholders})",
                            [row[c] for c in cols],
                        )
                    total += 1
            log_metadata(conn, rel, "alignment_runs", total)
            continue

        for run in runs:
            cond = run.get("condition", "")
            seed = run.get("seed", 0)
            qs = run.get("quick_scores", {})
            qd = run.get("quick_deltas", {})

            row = {
                "source_file": rel,
                "model": model,
                "backend": backend,
                "experiment": experiment,
                "condition": cond,
                "seed": seed,
                "rho_weight": config.get("rho_weight", run.get("rho_weight")),
                "sft_size": config.get("sft_size"),
                "lora_rank": config.get("lora_rank"),
                "epochs": config.get("epochs"),
                "lr": config.get("lr"),
                "margin": config.get("margin"),
                "elapsed_seconds": run.get("elapsed_seconds"),
                "train_ce_loss": run.get("train_ce_loss"),
                "train_rho_loss": run.get("train_rho_loss"),
                "train_steps": run.get("train_steps"),
            }
            for b in BEHAVIORS:
                row[f"bl_{b}"] = bl.get(b)
                row[b] = qs.get(b)
                row[f"d_{b}"] = qd.get(b)

            if not dry_run:
                cols = list(row.keys())
                placeholders = ", ".join(["?"] * len(cols))
                conn.execute(
                    f"INSERT INTO alignment_runs ({', '.join(cols)}) VALUES ({placeholders})",
                    [row[c] for c in cols],
                )
            total += 1

        if runs:
            log_metadata(conn, rel, "alignment_runs", len(runs))

    conn.commit()
    print(f"  alignment_runs: {total} rows")


def ingest_hybrid_sweep(conn: sqlite3.Connection, dry_run: bool = False):
    """Ingest results/hybrid_sweep/*/hybrid_result.json into hybrid_sweep."""
    sweep_dir = RESULTS_DIR / "hybrid_sweep"
    if not sweep_dir.exists():
        return

    total = 0
    for config_dir in sorted(sweep_dir.iterdir()):
        result_path = config_dir / "hybrid_result.json"
        if not result_path.exists():
            continue

        data = load_json(result_path)
        rel = str(result_path.relative_to(RESULTS_DIR))
        cfg = data.get("config", {})
        before = data.get("audit_before", {})
        after = data.get("audit_after", {})

        row = {
            "source_file": rel,
            "model_name": data.get("model_name", ""),
            "config_tag": config_dir.name,
            "timestamp": data.get("timestamp", ""),
            "compress_ratio": cfg.get("compress_ratio"),
            "freeze_fraction": cfg.get("freeze_fraction"),
            "sae_layer": cfg.get("sae_layer"),
            "rho_weight": cfg.get("rho_weight"),
            "seed": cfg.get("seed"),
            "total_elapsed_sec": data.get("total_elapsed_sec"),
        }

        before_vals = []
        after_vals = []
        for b in BEHAVIORS:
            bv = before.get(b)
            av = after.get(b)
            row[f"before_{b}"] = bv
            row[f"after_{b}"] = av
            row[f"d_{b}"] = (av - bv) if av is not None and bv is not None else None
            if bv is not None:
                before_vals.append(bv)
            if av is not None:
                after_vals.append(av)

        row["mean_before"] = sum(before_vals) / len(before_vals) if before_vals else None
        row["mean_after"] = sum(after_vals) / len(after_vals) if after_vals else None
        if row["mean_before"] is not None and row["mean_after"] is not None:
            row["mean_delta"] = row["mean_after"] - row["mean_before"]
        else:
            row["mean_delta"] = None

        if not dry_run:
            cols = list(row.keys())
            placeholders = ", ".join(["?"] * len(cols))
            conn.execute(
                f"INSERT INTO hybrid_sweep ({', '.join(cols)}) VALUES ({placeholders})",
                [row[c] for c in cols],
            )
        total += 1

    conn.commit()
    print(f"  hybrid_sweep: {total} rows")


def ingest_steering(conn: sqlite3.Connection, dry_run: bool = False):
    """Ingest results/steering/heatmap_*.json into steering_heatmap."""
    steering_dir = RESULTS_DIR / "steering"
    if not steering_dir.exists():
        return

    total = 0
    for fpath in sorted(steering_dir.glob("heatmap_*.json")):
        data = load_json(fpath)
        rel = str(fpath.relative_to(RESULTS_DIR))
        model_id = data.get("model_id", "")
        experiment = data.get("experiment", "")
        baselines = data.get("baselines", {})
        norms = data.get("vector_norms", {})

        for entry in data.get("sweep", []):
            layer = entry.get("layer")
            depth = entry.get("depth_pct")
            alpha = entry.get("alpha", data.get("alpha"))
            elapsed = entry.get("elapsed_s")
            results = entry.get("results", {})
            norm = norms.get(str(layer))

            for beh_name, beh_data in results.items():
                if not isinstance(beh_data, dict):
                    continue
                rho_steered = beh_data.get("rho")
                bl_data = baselines.get(beh_name, {})
                rho_bl = bl_data.get("rho") if isinstance(bl_data, dict) else None

                row = {
                    "source_file": rel,
                    "model_id": model_id,
                    "experiment": experiment,
                    "layer": layer,
                    "depth_pct": depth,
                    "alpha": alpha,
                    "vector_norm": norm,
                    "elapsed_seconds": elapsed,
                    "behavior": beh_name,
                    "rho_baseline": rho_bl,
                    "rho_steered": rho_steered,
                    "rho_delta": (rho_steered - rho_bl) if rho_steered is not None and rho_bl is not None else None,
                }

                if not dry_run:
                    cols = list(row.keys())
                    placeholders = ", ".join(["?"] * len(cols))
                    conn.execute(
                        f"INSERT INTO steering_heatmap ({', '.join(cols)}) VALUES ({placeholders})",
                        [row[c] for c in cols],
                    )
                total += 1

    conn.commit()
    print(f"  steering_heatmap: {total} rows")


def ingest_attack_defense(conn: sqlite3.Connection, dry_run: bool = False):
    """Ingest results/attack_defense/steering_sweep.json into attack_defense."""
    sweep_path = RESULTS_DIR / "attack_defense" / "steering_sweep.json"
    if not sweep_path.exists():
        return

    data = load_json(sweep_path)
    rel = str(sweep_path.relative_to(RESULTS_DIR))
    model = data.get("model", "")
    layer = data.get("layer", 0)

    total = 0
    for beh_name in ["refusal", "deception"]:
        beh = data.get(beh_name, {})
        if not beh:
            continue
        status = beh.get("status", "")
        n_features = beh.get("n_features", 0)

        for result in beh.get("results", []):
            row = {
                "source_file": rel,
                "model": model,
                "layer": layer,
                "behavior": beh_name,
                "status": status,
                "n_features": n_features,
                "scale": result.get("scale"),
                "rho": result.get("rho"),
                "delta": result.get("delta"),
                "n_regressed": result.get("n_regressed"),
                "mean_collateral": result.get("mean_collateral"),
            }

            if not dry_run:
                cols = list(row.keys())
                placeholders = ", ".join(["?"] * len(cols))
                conn.execute(
                    f"INSERT INTO attack_defense ({', '.join(cols)}) VALUES ({placeholders})",
                    [row[c] for c in cols],
                )
            total += 1

    conn.commit()
    print(f"  attack_defense: {total} rows")


def ingest_freeze_sweep(conn: sqlite3.Connection, dry_run: bool = False):
    """Ingest results/freeze_sweep/sweep_v2.json into freeze_sweep."""
    for fname in ["sweep_v2.json", "sweep.json"]:
        fpath = RESULTS_DIR / "freeze_sweep" / fname
        if not fpath.exists():
            continue

        data = load_json(fpath)
        rel = str(fpath.relative_to(RESULTS_DIR))
        total = 0

        for model_key, entry in data.items():
            if not isinstance(entry, dict):
                continue
            model = entry.get("model", model_key)
            model_id = entry.get("model_id", "")
            cr = entry.get("compress_ratio", 1.0)
            fr = entry.get("freeze_ratio", 0.0)
            behaviors = entry.get("behaviors", {})

            for beh_name, beh_data in behaviors.items():
                if not isinstance(beh_data, dict):
                    continue

                # Handle baseline vs compressed structure
                if "baseline" in beh_data and "compressed" in beh_data:
                    bl = beh_data["baseline"]
                    cp = beh_data["compressed"]
                    rho_bl = bl.get("rho") if isinstance(bl, dict) else None
                    rho_cp = cp.get("rho") if isinstance(cp, dict) else None
                    retention = cp.get("retention") if isinstance(cp, dict) else None
                    pos = cp.get("positive_count") if isinstance(cp, dict) else None
                    tot = cp.get("total") if isinstance(cp, dict) else None
                else:
                    rho_bl = beh_data.get("rho")
                    rho_cp = rho_bl
                    retention = beh_data.get("retention")
                    pos = beh_data.get("positive_count")
                    tot = beh_data.get("total")

                row = {
                    "source_file": rel,
                    "model": model,
                    "model_id": model_id,
                    "compress_ratio": cr,
                    "freeze_ratio": fr,
                    "behavior": beh_name,
                    "rho_baseline": rho_bl,
                    "rho_compressed": rho_cp,
                    "rho_delta": (rho_cp - rho_bl) if rho_cp is not None and rho_bl is not None else None,
                    "retention": retention,
                    "positive_count": pos,
                    "total": tot,
                }

                if not dry_run:
                    cols = list(row.keys())
                    placeholders = ", ".join(["?"] * len(cols))
                    conn.execute(
                        f"INSERT INTO freeze_sweep ({', '.join(cols)}) VALUES ({placeholders})",
                        [row[c] for c in cols],
                    )
                total += 1

        log_metadata(conn, rel, "freeze_sweep", total)
        conn.commit()
        print(f"  freeze_sweep ({fname}): {total} rows")


def ingest_leaderboard(conn: sqlite3.Connection, dry_run: bool = False):
    """Ingest results/leaderboard/merged_audit.json into leaderboard."""
    fpath = RESULTS_DIR / "leaderboard" / "merged_audit.json"
    if not fpath.exists():
        return

    data = load_json(fpath)
    rel = str(fpath.relative_to(RESULTS_DIR))
    total = 0

    for model_key, entry in data.items():
        if not isinstance(entry, dict):
            continue

        model_id = entry.get("model_id", "")
        model_type = entry.get("model_type", "")
        description = entry.get("description", "")
        device = entry.get("device", "")
        timestamp = entry.get("timestamp", "")
        behaviors = entry.get("behaviors", {})

        for beh_name, beh_data in behaviors.items():
            if not isinstance(beh_data, dict):
                continue

            row = {
                "source_file": rel,
                "model_key": model_key,
                "model_id": model_id,
                "model_type": model_type,
                "description": description,
                "device": device,
                "timestamp": timestamp,
                "behavior": beh_name,
                "rho": beh_data.get("rho"),
                "retention": beh_data.get("retention"),
                "positive_count": beh_data.get("positive_count"),
                "total": beh_data.get("total"),
            }

            if not dry_run:
                cols = list(row.keys())
                placeholders = ", ".join(["?"] * len(cols))
                conn.execute(
                    f"INSERT INTO leaderboard ({', '.join(cols)}) VALUES ({placeholders})",
                    [row[c] for c in cols],
                )
            total += 1

    conn.commit()
    print(f"  leaderboard: {total} rows")


def ingest_fidelity_bench(conn: sqlite3.Connection, dry_run: bool = False):
    """Ingest results/fidelity_bench_*.json into fidelity_bench."""
    total = 0
    for fpath in sorted(RESULTS_DIR.glob("fidelity_bench_*.json")):
        data = load_json(fpath)
        rel = str(fpath.relative_to(RESULTS_DIR))
        model = data.get("model", "")
        elapsed = data.get("elapsed_seconds")

        for cat in data.get("categories", []):
            row = {
                "source_file": rel,
                "model": model,
                "category": cat.get("category", ""),
                "n_probes": cat.get("n_probes"),
                "rho": cat.get("rho"),
                "rho_p": cat.get("rho_p"),
                "mean_delta": cat.get("mean_delta"),
                "accuracy": cat.get("accuracy"),
                "elapsed_seconds": elapsed,
            }

            if not dry_run:
                cols = list(row.keys())
                placeholders = ", ".join(["?"] * len(cols))
                conn.execute(
                    f"INSERT INTO fidelity_bench ({', '.join(cols)}) VALUES ({placeholders})",
                    [row[c] for c in cols],
                )
            total += 1

    conn.commit()
    print(f"  fidelity_bench: {total} rows")


def ingest_cf90_multiseed(conn: sqlite3.Connection, dry_run: bool = False):
    """Ingest results/cf90_multiseed_*.json into cf90_multiseed."""
    total = 0
    for fpath in sorted(RESULTS_DIR.glob("cf90_multiseed_*.json")):
        data = load_json(fpath)
        rel = str(fpath.relative_to(RESULTS_DIR))
        model = data.get("model", "")
        ratio = data.get("ratio")

        for seed_data in data.get("seeds", []):
            row = {
                "source_file": rel,
                "model": model,
                "ratio": ratio,
                "seed": seed_data.get("seed"),
                "retention": seed_data.get("retention"),
                "rho_before": seed_data.get("rho_before"),
                "rho_after": seed_data.get("rho_after"),
                "rho_drop": seed_data.get("rho_drop"),
                "n_compressed": seed_data.get("n_compressed"),
                "elapsed_seconds": seed_data.get("elapsed"),
            }

            if not dry_run:
                cols = list(row.keys())
                placeholders = ", ".join(["?"] * len(cols))
                conn.execute(
                    f"INSERT INTO cf90_multiseed ({', '.join(cols)}) VALUES ({placeholders})",
                    [row[c] for c in cols],
                )
            total += 1

    conn.commit()
    print(f"  cf90_multiseed: {total} rows")


def main():
    parser = argparse.ArgumentParser(description="Build master SQLite from result JSONs.")
    parser.add_argument("-o", "--output", type=str, default=str(DEFAULT_DB))
    parser.add_argument("--dry-run", action="store_true", help="Count rows without writing")
    args = parser.parse_args()

    db_path = Path(args.output)
    if db_path.exists() and not args.dry_run:
        db_path.unlink()
        print(f"Removed existing {db_path}")

    print(f"\nBuilding master database: {db_path}")
    print(f"Source: {RESULTS_DIR}")
    print(f"{'='*60}\n")

    conn = sqlite3.connect(str(db_path)) if not args.dry_run else sqlite3.connect(":memory:")
    create_tables(conn)

    ingest_alignment(conn, dry_run=args.dry_run)
    ingest_hybrid_sweep(conn, dry_run=args.dry_run)
    ingest_steering(conn, dry_run=args.dry_run)
    ingest_attack_defense(conn, dry_run=args.dry_run)
    ingest_freeze_sweep(conn, dry_run=args.dry_run)
    ingest_leaderboard(conn, dry_run=args.dry_run)
    ingest_fidelity_bench(conn, dry_run=args.dry_run)
    ingest_cf90_multiseed(conn, dry_run=args.dry_run)

    # Summary
    print(f"\n{'='*60}")
    c = conn.cursor()
    for table in ["alignment_runs", "hybrid_sweep", "steering_heatmap",
                   "attack_defense", "freeze_sweep", "leaderboard",
                   "fidelity_bench", "cf90_multiseed"]:
        c.execute(f"SELECT COUNT(*) FROM {table}")
        count = c.fetchone()[0]
        if count > 0:
            print(f"  {table:<25s} {count:>6} rows")

    c.execute("SELECT COUNT(*) FROM metadata")
    print(f"\n  Total source files tracked: {c.fetchone()[0]}")

    if not args.dry_run:
        # File size
        size = db_path.stat().st_size
        if size > 1_000_000:
            print(f"  Database size: {size / 1_000_000:.1f} MB")
        else:
            print(f"  Database size: {size / 1_000:.1f} KB")

    conn.close()
    print(f"\nDone.")


if __name__ == "__main__":
    main()
