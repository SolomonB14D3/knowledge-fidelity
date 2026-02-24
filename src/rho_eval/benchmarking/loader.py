"""Dataset versioning and probe loading for Fidelity-Bench.

Handles version tracking so benchmark results are comparable over time.
Each certificate embeds a probe hash for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from .schema import BENCHMARK_VERSION, FidelityCertificate


# ── Constants ──────────────────────────────────────────────────────────────

BENCH_VERSION = BENCHMARK_VERSION

# Bench probe data lives alongside other probes
_BENCH_DATA_DIR = Path(__file__).resolve().parent.parent / "probes" / "data" / "bench"

DOMAIN_FILES = {
    "logic": "logic.json",
    "social": "social.json",
    "clinical": "clinical.json",
}

DOMAIN_DEFAULTS = {
    "logic": 40,
    "social": 40,
    "clinical": 40,
}


# ── Probe Hash ─────────────────────────────────────────────────────────────

def compute_probe_hash(probes: Optional[list[dict]] = None) -> str:
    """Compute deterministic SHA256 hash of probe data.

    If probes is None, computes hash from the shipped bench probe files.

    Args:
        probes: List of probe dicts, or None to hash shipped files.

    Returns:
        Hex digest of SHA256 hash.
    """
    if probes is not None:
        # Sort for determinism, then hash the JSON
        canonical = json.dumps(
            sorted(probes, key=lambda p: p.get("id", "")),
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    # Hash all shipped bench files
    h = hashlib.sha256()
    for domain in sorted(DOMAIN_FILES.keys()):
        fpath = _BENCH_DATA_DIR / DOMAIN_FILES[domain]
        if fpath.exists():
            h.update(fpath.read_bytes())
    return h.hexdigest()


# ── Probe Loading ──────────────────────────────────────────────────────────

def load_bench_probes(
    domain: str,
    n: Optional[int] = None,
    seed: int = 42,
) -> list[dict]:
    """Load probes for a specific benchmark domain.

    Args:
        domain: One of "logic", "social", "clinical".
        n: Number of probes to return (None → all).
        seed: Random seed for subsampling.

    Returns:
        List of probe dicts with "text", "false", "id", "domain" keys.

    Raises:
        ValueError: If domain is unknown.
        FileNotFoundError: If probe data file is missing.
    """
    import random

    if domain not in DOMAIN_FILES:
        raise ValueError(
            f"Unknown domain: {domain!r}. "
            f"Available: {list(DOMAIN_FILES.keys())}"
        )

    fpath = _BENCH_DATA_DIR / DOMAIN_FILES[domain]
    if not fpath.exists():
        raise FileNotFoundError(
            f"Bench probe data not found: {fpath}\n"
            f"Expected probe JSON at: {_BENCH_DATA_DIR}"
        )

    with open(fpath) as f:
        probes = json.load(f)

    if n is not None and n < len(probes):
        rng = random.Random(seed)
        probes = rng.sample(probes, n)

    return probes


def load_all_bench_probes(
    domains: Optional[list[str]] = None,
    n_per_domain: Optional[int] = None,
    seed: int = 42,
) -> list[dict]:
    """Load probes from all benchmark domains.

    Args:
        domains: List of domains to load (None → all).
        n_per_domain: Probes per domain (None → all).
        seed: Random seed.

    Returns:
        Combined list of probes from all requested domains.
    """
    if domains is None:
        domains = list(DOMAIN_FILES.keys())

    all_probes = []
    for domain in domains:
        probes = load_bench_probes(domain, n=n_per_domain, seed=seed)
        all_probes.extend(probes)

    return all_probes


# ── Version Validation ─────────────────────────────────────────────────────

def validate_version(certificate: FidelityCertificate) -> bool:
    """Check that a certificate was produced with the current probe set.

    Args:
        certificate: A loaded FidelityCertificate.

    Returns:
        True if probe hashes match.
    """
    current_hash = compute_probe_hash()
    return certificate.probe_hash == current_hash


def get_bench_metadata() -> dict:
    """Return metadata about the current benchmark probe set.

    Returns:
        Dict with version, probe_hash, n_probes, domains, etc.
    """
    n_total = 0
    domain_counts = {}

    for domain, filename in DOMAIN_FILES.items():
        fpath = _BENCH_DATA_DIR / filename
        if fpath.exists():
            with open(fpath) as f:
                probes = json.load(f)
            count = len(probes)
        else:
            count = 0
        domain_counts[domain] = count
        n_total += count

    return {
        "version": BENCH_VERSION,
        "probe_hash": compute_probe_hash(),
        "n_probes": n_total,
        "domains": list(DOMAIN_FILES.keys()),
        "domain_counts": domain_counts,
        "data_dir": str(_BENCH_DATA_DIR),
    }
