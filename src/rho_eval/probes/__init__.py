"""Probe data package — shipped JSON files, discovery API, and legacy probe API.

Probe data lives in probes/data/{behavior}/*.json. This package provides
a simple API to discover and load them without hardcoding file paths.

Also re-exports all legacy functions from the old probes.py module for
backward compatibility (get_default_probes, get_importance_prompts, etc.).

Usage (new API):
    from rho_eval.probes import list_probe_sets, get_probes

    list_probe_sets()
    probes = get_probes("factual/default")

Usage (legacy API, still works):
    from rho_eval.probes import get_default_probes, get_importance_prompts
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

from ..utils import PROBES_DIR

# New probe registry API
from .registry import list_probe_sets, get_probes, get_probe_counts, PROBE_DATA_DIR


# ═══════════════════════════════════════════════════════════════════════════
# Legacy API (backward compatible with old probes.py)
# ═══════════════════════════════════════════════════════════════════════════

# Default probes serve dual purpose:
# - As prompts for gradient-based importance scoring (SVD)
# - As true/false pairs for confidence ratio measurement (cartography)

DEFAULT_PROBES = [
    # --- Geography ---
    {"text": "The capital of France is Paris.",
     "false": "The capital of France is Berlin.",
     "domain": "geography", "id": "capital_france"},
    {"text": "The capital of Japan is Tokyo.",
     "false": "The capital of Japan is Beijing.",
     "domain": "geography", "id": "capital_japan"},
    {"text": "The capital of Australia is Canberra.",
     "false": "The capital of Australia is Sydney.",
     "domain": "geography", "id": "capital_australia"},
    {"text": "The largest ocean is the Pacific Ocean.",
     "false": "The largest ocean is the Atlantic Ocean.",
     "domain": "geography", "id": "largest_ocean"},
    {"text": "Mount Everest is the tallest mountain on Earth.",
     "false": "Mount Kilimanjaro is the tallest mountain on Earth.",
     "domain": "geography", "id": "tallest_mountain"},

    # --- Science ---
    {"text": "Water boils at 100 degrees Celsius at sea level.",
     "false": "Water boils at 50 degrees Celsius at sea level.",
     "domain": "science", "id": "water_boiling"},
    {"text": "The Earth orbits the Sun.",
     "false": "The Sun orbits the Earth.",
     "domain": "science", "id": "earth_orbit"},
    {"text": "Light travels faster than sound.",
     "false": "Sound travels faster than light.",
     "domain": "science", "id": "light_vs_sound"},
    {"text": "Diamonds are made of carbon.",
     "false": "Diamonds are made of silicon.",
     "domain": "science", "id": "diamond_composition"},
    {"text": "The chemical symbol for gold is Au.",
     "false": "The chemical symbol for gold is Ag.",
     "domain": "science", "id": "gold_symbol"},
    {"text": "Humans have 23 pairs of chromosomes.",
     "false": "Humans have 30 pairs of chromosomes.",
     "domain": "science", "id": "chromosomes"},
    {"text": "The speed of light is approximately 300,000 kilometers per second.",
     "false": "The speed of light is approximately 300,000 miles per second.",
     "domain": "science", "id": "speed_of_light"},

    # --- History ---
    {"text": "World War II ended in 1945.",
     "false": "World War II ended in 1952.",
     "domain": "history", "id": "ww2_end"},
    {"text": "The Berlin Wall fell in 1989.",
     "false": "The Berlin Wall fell in 1975.",
     "domain": "history", "id": "berlin_wall"},
    {"text": "Shakespeare wrote Hamlet.",
     "false": "Shakespeare wrote The Odyssey.",
     "domain": "history", "id": "shakespeare"},
    {"text": "The first moon landing was in 1969.",
     "false": "The first moon landing was in 1959.",
     "domain": "history", "id": "moon_landing"},

    # --- Biology ---
    {"text": "Humans have two lungs.",
     "false": "Humans have three lungs.",
     "domain": "biology", "id": "human_lungs"},
    {"text": "The heart pumps blood through the body.",
     "false": "The liver pumps blood through the body.",
     "domain": "biology", "id": "heart_function"},
    {"text": "DNA stands for deoxyribonucleic acid.",
     "false": "DNA stands for dinitrogen acid.",
     "domain": "biology", "id": "dna_name"},
    {"text": "Photosynthesis converts sunlight into chemical energy.",
     "false": "Photosynthesis converts moonlight into chemical energy.",
     "domain": "biology", "id": "photosynthesis"},
]


MANDELA_PROBES = [
    {"text": "The Berenstain Bears is a popular children's book series.",
     "false": "The Berenstein Bears is a popular children's book series.",
     "domain": "mandela", "id": "berenstain_bears",
     "note": "Commonly misremembered as 'Berenstein'"},
    {"text": 'In Star Wars, Darth Vader says "No, I am your father."',
     "false": 'In Star Wars, Darth Vader says "Luke, I am your father."',
     "domain": "mandela", "id": "vader_quote",
     "note": "Almost universally misquoted"},
    {"text": "The Monopoly Man does not wear a monocle.",
     "false": "The Monopoly Man wears a monocle.",
     "domain": "mandela", "id": "monopoly_monocle",
     "note": "Many people believe he wears one"},
    {"text": "Curious George does not have a tail.",
     "false": "Curious George has a long tail.",
     "domain": "mandela", "id": "curious_george_tail",
     "note": "Monkeys typically do, but CG doesn't"},
    {"text": "Fruit of the Loom logo does not contain a cornucopia.",
     "false": "Fruit of the Loom logo contains a cornucopia.",
     "domain": "mandela", "id": "fruit_loom_cornucopia",
     "note": "One of the strongest Mandela effects"},
    {"text": 'In Casablanca, the actual line is "Here\'s looking at you, kid."',
     "false": 'In Casablanca, the famous line is "Play it again, Sam."',
     "domain": "mandela", "id": "casablanca_quote",
     "note": "Play it again Sam is never said"},
]


MEDICAL_PROBES = [
    {"text": "Vaccines do not cause autism.",
     "false": "Vaccines cause autism.",
     "domain": "medical", "id": "vaccines_autism"},
    {"text": "Antibiotics do not work against viral infections.",
     "false": "Antibiotics work against viral infections.",
     "domain": "medical", "id": "antibiotics_viral"},
    {"text": "The flu vaccine cannot give you the flu.",
     "false": "The flu vaccine can give you the flu.",
     "domain": "medical", "id": "flu_vaccine"},
    {"text": "Cracking your knuckles does not cause arthritis.",
     "false": "Cracking your knuckles causes arthritis.",
     "domain": "medical", "id": "knuckle_cracking"},
    {"text": "Humans use all of their brain, not just 10 percent.",
     "false": "Humans only use 10 percent of their brain.",
     "domain": "medical", "id": "brain_10_percent"},
]


# ── Legacy functions ──────────────────────────────────────────────────────

def get_default_probes() -> list[dict]:
    """Return the built-in default probes."""
    return DEFAULT_PROBES.copy()


def get_mandela_probes() -> list[dict]:
    """Return the Mandela effect probes."""
    return MANDELA_PROBES.copy()


def get_medical_probes() -> list[dict]:
    """Return the medical claim probes."""
    return MEDICAL_PROBES.copy()


def get_commonsense_probes() -> list[dict]:
    """Return the commonsense myth probes (loaded from data/probes/)."""
    return load_probes(PROBES_DIR / "commonsense_myths.json")


def get_truthfulqa_probes() -> list[dict]:
    """Return the TruthfulQA-derived probes (loaded from data/probes/)."""
    return load_probes(PROBES_DIR / "truthfulqa_subset.json")


def get_all_probes() -> list[dict]:
    """Return all built-in probes (default + mandela + medical + commonsense + truthfulqa)."""
    all_probes = DEFAULT_PROBES + MANDELA_PROBES + MEDICAL_PROBES
    for loader in [get_commonsense_probes, get_truthfulqa_probes]:
        try:
            all_probes.extend(loader())
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    return all_probes


def load_probes(path: Union[str, Path]) -> list[dict]:
    """Load probes from a JSON file.

    Expected format: list of dicts with at least "text" and "false" keys.
    Optional: "domain", "id", "note".
    """
    with open(path) as f:
        probes = json.load(f)

    for i, p in enumerate(probes):
        if "text" not in p:
            raise ValueError(f"Probe {i} missing 'text' key: {p}")
        if "false" not in p:
            raise ValueError(f"Probe {i} missing 'false' key: {p}")
        p.setdefault("domain", "custom")
        p.setdefault("id", f"custom_{i}")

    return probes


def save_probes(probes: list[dict], path: Union[str, Path]) -> None:
    """Save probes to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(probes, f, indent=2)


def get_importance_prompts(
    probes: Optional[list[dict]] = None,
) -> list[str]:
    """Extract just the true-text prompts for SVD importance scoring."""
    if probes is None:
        probes = DEFAULT_PROBES
    return [p["text"] for p in probes]


__all__ = [
    # New API
    "list_probe_sets",
    "get_probes",
    "get_probe_counts",
    "PROBE_DATA_DIR",
    # Legacy API
    "DEFAULT_PROBES",
    "MANDELA_PROBES",
    "MEDICAL_PROBES",
    "get_default_probes",
    "get_mandela_probes",
    "get_medical_probes",
    "get_commonsense_probes",
    "get_truthfulqa_probes",
    "get_all_probes",
    "load_probes",
    "save_probes",
    "get_importance_prompts",
]
