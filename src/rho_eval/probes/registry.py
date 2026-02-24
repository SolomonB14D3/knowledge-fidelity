"""Probe set discovery and loading.

Discovers JSON files in probes/data/ and provides a clean API for loading
them. Supports both shipped probes and user-added custom probe sets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

PROBE_DATA_DIR = Path(__file__).resolve().parent / "data"


def list_probe_sets() -> list[str]:
    """Return all available probe set names.

    Names are relative paths without .json extension, e.g.:
        "factual/default", "toxicity/toxigen_200", "bias/bbq_300"

    Returns:
        Sorted list of probe set names.
    """
    if not PROBE_DATA_DIR.exists():
        return []

    sets = []
    for json_file in sorted(PROBE_DATA_DIR.rglob("*.json")):
        rel = json_file.relative_to(PROBE_DATA_DIR)
        name = str(rel.with_suffix(""))
        sets.append(name)
    return sorted(sets)


def get_probes(
    name: str,
    n: Optional[int] = None,
    seed: int = 42,
) -> list[dict]:
    """Load a probe set by name.

    Args:
        name: Probe set name (e.g., "factual/default", "bias/bbq_300").
              Can also be a full path or path to a JSON file.
        n: If provided and < len(data), subsample with seed.
        seed: Random seed for subsampling.

    Returns:
        List of probe dicts.

    Raises:
        FileNotFoundError: If probe set doesn't exist.
        ValueError: If probes are missing required keys.
    """
    # Try as a shipped probe set name first
    path = PROBE_DATA_DIR / f"{name}.json"
    if not path.exists():
        # Try as an absolute/relative path
        path = Path(name)
        if not path.exists():
            available = ", ".join(list_probe_sets())
            raise FileNotFoundError(
                f"Probe set '{name}' not found. Available: {available}"
            )

    with open(path) as f:
        probes = json.load(f)

    # Validate minimum required keys
    for i, p in enumerate(probes):
        if "text" not in p:
            raise ValueError(f"Probe {i} in '{name}' missing 'text' key")

    # Subsample if requested
    if n is not None and n < len(probes):
        import random
        rng = random.Random(seed)
        probes = rng.sample(probes, n)

    return probes


def get_probe_counts() -> dict[str, int]:
    """Return a dict of probe set names to their probe counts.

    Returns:
        Dict mapping probe set name to number of probes.
    """
    counts = {}
    for name in list_probe_sets():
        path = PROBE_DATA_DIR / f"{name}.json"
        try:
            with open(path) as f:
                data = json.load(f)
            counts[name] = len(data)
        except (json.JSONDecodeError, OSError):
            counts[name] = -1
    return counts
