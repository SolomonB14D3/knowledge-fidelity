#!/usr/bin/env python3
"""Generate NLI contrastive pairs from SNLI for reasoning injection.

Strategy (from plan):
  For each premise that has BOTH an entailment and a contradiction hypothesis:
    positive = premise + " " + entailment_hypothesis
    negative = premise + " " + contradiction_hypothesis

  Select top 300 by token overlap between positive and negative.
  This gives pairs where the same context leads to two different conclusions,
  matching the existing bias/sycophancy pair structure.

Usage:
    python scripts/generate_nli_pairs.py \
        --output data/contrastive/nli_pairs.json \
        --n 300
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from transformers import GPT2Tokenizer


def compute_token_overlap(text_a: str, text_b: str, enc) -> float:
    """Compute token overlap ratio between two texts."""
    tokens_a = set(enc.encode(text_a))
    tokens_b = set(enc.encode(text_b))
    if not tokens_a and not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def compute_char_overlap(text_a: str, text_b: str) -> float:
    """Compute character-level overlap."""
    common = sum(1 for a, b in zip(text_a, text_b) if a == b)
    max_len = max(len(text_a), len(text_b))
    return common / max_len if max_len > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Generate NLI contrastive pairs from SNLI")
    parser.add_argument("--output", type=str, default="data/contrastive/nli_pairs.json")
    parser.add_argument("--n", type=int, default=300, help="Number of pairs to select")
    parser.add_argument("--min-token-overlap", type=float, default=0.0,
                        help="Minimum token overlap ratio (0-1)")
    args = parser.parse_args()

    # Load SNLI
    print("Loading SNLI dataset...")
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/snli", split="train")
    print(f"  Total examples: {len(ds)}")

    # Group by premise — collect entailment and contradiction hypotheses
    # SNLI labels: 0=entailment, 1=neutral, 2=contradiction
    print("Grouping by premise...")
    premise_map = defaultdict(lambda: {"entailment": [], "contradiction": []})

    for row in ds:
        label = row["label"]
        premise = row["premise"]
        hypothesis = row["hypothesis"]

        if label == 0:  # entailment
            premise_map[premise]["entailment"].append(hypothesis)
        elif label == 2:  # contradiction
            premise_map[premise]["contradiction"].append(hypothesis)

    # Filter premises that have BOTH entailment and contradiction
    both = {p: v for p, v in premise_map.items()
            if v["entailment"] and v["contradiction"]}
    print(f"  Premises with entailment: {sum(1 for v in premise_map.values() if v['entailment'])}")
    print(f"  Premises with contradiction: {sum(1 for v in premise_map.values() if v['contradiction'])}")
    print(f"  Premises with BOTH: {len(both)}")

    # Generate all possible pairs from these premises
    print("Generating candidate pairs...")
    enc = GPT2Tokenizer.from_pretrained("gpt2")
    candidates = []

    for premise, hyps in both.items():
        for ent in hyps["entailment"]:
            for con in hyps["contradiction"]:
                # Skip degenerate pairs (identical entailment/contradiction)
                if ent.strip().lower() == con.strip().lower():
                    continue
                positive = f"{premise} {ent}"
                negative = f"{premise} {con}"
                # Skip if positive == negative after construction
                if positive == negative:
                    continue

                tok_overlap = compute_token_overlap(positive, negative, enc)
                char_overlap = compute_char_overlap(positive, negative)

                candidates.append({
                    "positive": positive,
                    "negative": negative,
                    "premise": premise,
                    "entailment": ent,
                    "contradiction": con,
                    "token_overlap": tok_overlap,
                    "char_overlap": char_overlap,
                })

    print(f"  Total candidate pairs: {len(candidates)}")

    # Filter by minimum overlap if specified
    if args.min_token_overlap > 0:
        candidates = [c for c in candidates if c["token_overlap"] >= args.min_token_overlap]
        print(f"  After min overlap filter ({args.min_token_overlap}): {len(candidates)}")

    # Sort by token overlap (descending) and take top N
    candidates.sort(key=lambda x: x["token_overlap"], reverse=True)
    selected = candidates[:args.n]

    print(f"\nSelected {len(selected)} pairs")
    overlaps = [s["token_overlap"] for s in selected]
    char_overlaps = [s["char_overlap"] for s in selected]
    print(f"  Token overlap: mean={sum(overlaps)/len(overlaps):.3f}, "
          f"min={min(overlaps):.3f}, max={max(overlaps):.3f}")
    print(f"  Char overlap:  mean={sum(char_overlaps)/len(char_overlaps):.3f}, "
          f"min={min(char_overlaps):.3f}, max={max(char_overlaps):.3f}")

    # Format for the contrastive training pipeline
    output_pairs = []
    for i, s in enumerate(selected):
        output_pairs.append({
            "positive": s["positive"],
            "negative": s["negative"],
            "id": f"nli_{i+1:03d}",
            "category": "nli_contradiction",
            "paradigm": "nli",
        })

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_pairs, indent=2))
    print(f"\nSaved {len(output_pairs)} pairs to {out_path}")

    # Print some examples
    print("\n--- Example pairs (top 5 by token overlap) ---")
    for s in selected[:5]:
        print(f"\n  Premise: {s['premise']}")
        print(f"  Entailment: {s['entailment']}")
        print(f"  Contradiction: {s['contradiction']}")
        print(f"  Token overlap: {s['token_overlap']:.3f}")

    print("\n--- Example pairs (bottom 5 of selected) ---")
    for s in selected[-5:]:
        print(f"\n  Premise: {s['premise']}")
        print(f"  Entailment: {s['entailment']}")
        print(f"  Contradiction: {s['contradiction']}")
        print(f"  Token overlap: {s['token_overlap']:.3f}")


if __name__ == "__main__":
    main()
