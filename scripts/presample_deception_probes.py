#!/usr/bin/env python3
"""Pre-sample deception probes from Anthropic HH-RLHF.

Extracts honest/deceptive text pairs from the chosen/rejected responses
in the HH-RLHF dataset. Chosen responses are honest/helpful; rejected
responses are evasive, misleading, or dishonest.

Filters:
  - Both texts must be > 50 characters
  - Length ratio must be < 3x (avoids trivially different lengths)
  - Both texts must contain the assistant turn
  - Deduplication by content hash

Output: src/rho_eval/probes/data/deception/hh_rlhf_100.json
"""

import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset


def extract_last_assistant_turn(text: str) -> str:
    """Extract the last assistant turn from an HH-RLHF conversation.

    HH-RLHF uses the format:
      Human: ...
      Assistant: ...
      Human: ...
      Assistant: ...

    We extract the final Assistant turn as the response text.
    """
    parts = text.split("\n\nAssistant: ")
    if len(parts) < 2:
        return ""
    return parts[-1].strip()


def build_deception_probes(n: int = 100, seed: int = 42) -> list[dict]:
    """Build deception probe pairs from HH-RLHF.

    Filters for pairs where chosen (honest) and rejected (deceptive)
    responses clearly differ on a helpfulness/honesty axis.

    Args:
        n: Number of probe pairs to extract.
        seed: Random seed for reproducible sampling.

    Returns:
        List of probe dicts with keys: honest, deceptive, id, category.
    """
    rng = random.Random(seed)

    # Load the helpful-base split (not harmless -- we want the
    # honest-vs-unhelpful signal, not the safe-vs-unsafe signal)
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train")
    print(f"Loaded {len(ds)} examples from HH-RLHF helpful-base")

    candidates = []
    seen_hashes = set()

    for ex in ds:
        chosen_text = extract_last_assistant_turn(ex["chosen"])
        rejected_text = extract_last_assistant_turn(ex["rejected"])

        # Filter: both must have content
        if not chosen_text or not rejected_text:
            continue

        # Filter: minimum length
        if len(chosen_text) < 50 or len(rejected_text) < 50:
            continue

        # Filter: length ratio (avoid trivially different lengths)
        ratio = max(len(chosen_text), len(rejected_text)) / min(len(chosen_text), len(rejected_text))
        if ratio > 3.0:
            continue

        # Filter: must be meaningfully different (not just rephrasing)
        if chosen_text[:50] == rejected_text[:50]:
            continue

        # Dedup by content hash
        h = hashlib.md5((chosen_text[:100] + rejected_text[:100]).encode()).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        # Truncate very long texts to keep probe file reasonable
        chosen_text = chosen_text[:500]
        rejected_text = rejected_text[:500]

        # Extract the human question for categorization
        human_parts = ex["chosen"].split("\n\nHuman: ")
        question = human_parts[-1].split("\n\nAssistant:")[0].strip() if len(human_parts) > 1 else ""

        # Simple category heuristic
        q_lower = question.lower()
        if any(w in q_lower for w in ["code", "program", "python", "function", "api"]):
            category = "coding"
        elif any(w in q_lower for w in ["math", "calculate", "equation", "number"]):
            category = "math"
        elif any(w in q_lower for w in ["explain", "what is", "how does", "why"]):
            category = "explanation"
        elif any(w in q_lower for w in ["write", "essay", "story", "poem"]):
            category = "creative"
        elif any(w in q_lower for w in ["advice", "should i", "recommend", "suggest"]):
            category = "advice"
        else:
            category = "general"

        candidates.append({
            "honest": chosen_text,
            "deceptive": rejected_text,
            "category": category,
            "question_preview": question[:100],
        })

    print(f"Found {len(candidates)} valid candidates after filtering")

    # Sample n probes
    if len(candidates) > n:
        candidates = rng.sample(candidates, n)

    # Assign IDs
    probes = []
    for i, c in enumerate(candidates):
        probes.append({
            "honest": c["honest"],
            "deceptive": c["deceptive"],
            "id": f"hhrlhf_{i:03d}",
            "category": c["category"],
        })

    return probes


def main():
    output_path = Path(__file__).parent.parent / "src" / "rho_eval" / "probes" / "data" / "deception" / "hh_rlhf_100.json"

    probes = build_deception_probes(n=100, seed=42)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(probes, f, indent=2)

    print(f"Wrote {len(probes)} deception probes to {output_path}")

    # Print category distribution
    cats = {}
    for p in probes:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    print("Category distribution:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
