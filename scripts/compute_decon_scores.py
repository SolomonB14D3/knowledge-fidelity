#!/usr/bin/env python3
"""Compute deconcentration scores from existing subspace_report.json files.

Deconcentration score measures the ratio of SV1 reduction to dimensionality
expansion relative to the vanilla baseline:

    decon = (1 - sv1_post / sv1_van) * (eff_post / eff_van)

Positive decon → productive restructuring (SV1 drops, eff_dim expands).
Near-zero/negative → inflationary noise (SV1 stays/grows, eff_dim stuck at 1).

Output: results/developmental_sweep/decon_scores.json
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCALE_DIR = ROOT / "results" / "scale_ladder"
OUT_DIR = ROOT / "results" / "developmental_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def decon_score(sv1_post, sv1_van, eff_post, eff_van):
    """Compute deconcentration score."""
    if sv1_van == 0:
        return 0.0
    return (1.0 - sv1_post / sv1_van) * (eff_post / eff_van)


def load_bias_subspace(checkpoint_dir):
    """Extract bias SV1 and eff_dim from the last layer of a subspace_report."""
    path = checkpoint_dir / "subspace_report.json"
    with open(path) as f:
        data = json.load(f)
    n_layers = data["n_layers"]
    last_layer = str(n_layers - 1)
    bias_data = data["effective_dimensionality"]["bias"][last_layer]
    sv1 = bias_data["singular_values_top5"][0]
    eff_dim = bias_data["effective_dim"]
    return sv1, eff_dim


def load_bias_rho(checkpoint_dir):
    """Extract bias rho from audit_report.json."""
    path = checkpoint_dir / "audit_report.json"
    with open(path) as f:
        data = json.load(f)
    for result in data["results"]:
        if result["behavior"] == "bias":
            return result["rho"]
    return 0.0


# ── Define conditions ──────────────────────────────────────────────────
CONDITIONS = [
    # (label, scale, dir_name, type)
    # Direct injection
    ("Bias-only 3M", "3M", "3M_seed42_contr_bia_r20", "direct"),
    ("Bias-only 5M", "5M", "5M_seed42_contr_bia_r20", "direct"),
    ("Bias-only 7M", "7M", "7M_seed42_contr_bia_r20", "direct"),
    # Cross-transfer
    ("Syco-only 3M", "3M", "3M_seed42_contr_syc_r20", "cross"),
    ("Syco-only 5M", "5M", "5M_seed42_contr_syc_r20", "cross"),
    ("Syco-only 7M", "7M", "7M_seed42_contr_syc_r20", "cross"),
    # Null conditions
    ("NLI 3M", "3M", "3M_seed42_contr_nli_r20", "null"),
    ("Calculator 3M", "3M", "3M_seed42_contr_cal_r20", "null"),
    ("Subitizing 3M", "3M", "3M_seed42_contr_sub_r20", "null"),
    ("Primitive 3M", "3M", "3M_seed42_contr_pri_r20", "null"),
    ("NLI 7M", "7M", "7M_seed42_contr_nli_r20", "null"),
    ("Toxicity 7M", "7M", "7M_seed42_contr_tox_r20", "null"),
]

VANILLA = {
    "3M": "3M_seed42",
    "5M": "5M_seed42",
    "7M": "7M_seed42",
}

# ── Compute ────────────────────────────────────────────────────────────
results = []
van_cache = {}

for label, scale, dirname, ctype in CONDITIONS:
    # Load vanilla baseline (cached)
    if scale not in van_cache:
        van_dir = SCALE_DIR / VANILLA[scale]
        sv1_van, eff_van = load_bias_subspace(van_dir)
        van_cache[scale] = (sv1_van, eff_van)
    sv1_van, eff_van = van_cache[scale]

    # Load condition
    cond_dir = SCALE_DIR / dirname
    sv1_post, eff_post = load_bias_subspace(cond_dir)
    bias_rho = load_bias_rho(cond_dir)

    # Compute decon score
    # Note: when eff_van == 1 (all vanilla baselines), the formula simplifies
    # to (1 - sv1_post/sv1_van) * eff_post
    ds = decon_score(sv1_post, sv1_van, eff_post, eff_van)

    entry = {
        "label": label,
        "scale": scale,
        "type": ctype,
        "sv1_vanilla": round(sv1_van, 4),
        "sv1_post": round(sv1_post, 4),
        "eff_dim_vanilla": eff_van,
        "eff_dim_post": eff_post,
        "sv1_change_pct": round((sv1_post / sv1_van - 1) * 100, 1),
        "decon_score": round(ds, 4),
        "bias_rho": round(bias_rho, 4),
    }
    results.append(entry)
    print(f"{label:20s}  decon={ds:+.4f}  ρ={bias_rho:.4f}  "
          f"SV1: {sv1_van:.1f}→{sv1_post:.1f} ({(sv1_post/sv1_van-1)*100:+.1f}%)  "
          f"eff: {eff_van}→{eff_post}")

# ── Save ───────────────────────────────────────────────────────────────
output = {
    "description": "Deconcentration scores for all contrastive injection conditions",
    "formula": "decon = (1 - sv1_post / sv1_van) * (eff_post / eff_van)",
    "subspace": "bias",
    "layer": "last",
    "vanilla_baselines": {
        scale: {"sv1": round(v[0], 4), "eff_dim": v[1]}
        for scale, v in van_cache.items()
    },
    "conditions": results,
}

out_path = OUT_DIR / "decon_scores.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {out_path}")
