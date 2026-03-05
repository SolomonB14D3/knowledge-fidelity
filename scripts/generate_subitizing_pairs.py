#!/usr/bin/env python3
"""Generate 300 subitizing contrastive pairs for pretraining injection.

Subitizing = instantly recognizing the count of a small set of objects (1-5).
Positive = correct count, Negative = off-by-1-or-2 wrong count.
Token overlap between positive/negative is >90% (only the count token differs).

Output: data/contrastive/subitizing_pairs.json
"""

import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(42)

# ---------------------------------------------------------------------------
# Noun pool (50+ common nouns across categories)
# ---------------------------------------------------------------------------
ANIMALS = [
    "dog", "cat", "bird", "fish", "rabbit", "horse", "duck", "frog",
    "mouse", "turtle", "bear", "snake", "owl", "sheep", "cow",
]
HOUSEHOLD = [
    "cup", "plate", "spoon", "fork", "knife", "bowl", "glass", "lamp",
    "chair", "table", "pillow", "blanket", "clock", "mirror", "vase",
]
FOODS = [
    "apple", "banana", "orange", "cookie", "cake", "egg", "grape",
    "lemon", "muffin", "pear",
]
TOYS = [
    "ball", "block", "doll", "kite", "drum", "puzzle", "marble",
]
CLOTHING = [
    "hat", "shoe", "sock", "glove", "scarf", "boot", "belt",
]
TOOLS = [
    "hammer", "wrench", "brush", "bucket", "nail", "bolt",
]

ALL_NOUNS = ANIMALS + HOUSEHOLD + FOODS + TOYS + CLOTHING + TOOLS

ADJECTIVES = [
    "red", "blue", "green", "small", "big", "old", "new", "bright",
    "shiny", "wooden", "plastic", "soft", "round", "tiny", "large",
]

# ---------------------------------------------------------------------------
# Sentence frames — at least 5 per pattern
# ---------------------------------------------------------------------------
# {obj} will be substituted with the object phrase (with article).
# {N} will be substituted with the count.

FRAMES_SINGLE = [
    "{obj}. Count: {N}",
    "One {bare} is here. Count: {N}",
    "There is {obj}. Count: {N}",
    "A single {bare}. Count: {N}",
    "I see {obj}. Count: {N}",
    "Here is {obj}. Count: {N}",
    "Just {obj}. Count: {N}",
]

FRAMES_TWO = [
    "{obj1} and {obj2}. Count: {N}",
    "There is {obj1} and {obj2}. Count: {N}",
    "I see {obj1} and {obj2}. Count: {N}",
    "{obj1} next to {obj2}. Count: {N}",
    "Here are {obj1} and {obj2}. Count: {N}",
    "{obj1} with {obj2}. Count: {N}",
]

FRAMES_THREE_PLUS = [
    "{items}. Count: {N}",
    "There are {items}. Count: {N}",
    "I see {items}. Count: {N}",
    "Here are {items}. Count: {N}",
    "Objects: {items}. Count: {N}",
    "On the table: {items}. Count: {N}",
]


def vowel_start(word: str) -> bool:
    return word[0].lower() in "aeiou"


def article_for(word: str) -> str:
    return "an" if vowel_start(word) else "a"


def make_object_phrase(noun: str, use_adj: bool) -> tuple[str, str]:
    """Return (full phrase with article, bare noun/adj+noun)."""
    if use_adj:
        adj = random.choice(ADJECTIVES)
        bare = f"{adj} {noun}"
    else:
        bare = noun
    art = article_for(bare)
    return f"{art} {bare}", bare


def pick_wrong_count(correct: int) -> int:
    """Return a wrong count that differs by 1 or 2, stays in [1, 7], never equals correct."""
    offsets = []
    for d in [-2, -1, 1, 2]:
        candidate = correct + d
        if 1 <= candidate <= 7:
            offsets.append(candidate)
    return random.choice(offsets)


def pick_distinct_nouns(n: int) -> list[str]:
    """Pick n distinct nouns from the pool."""
    return random.sample(ALL_NOUNS, n)


def format_item_list(phrases: list[str]) -> str:
    """Oxford comma list: 'a, b, and c'."""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + ", and " + phrases[-1]


def should_use_adj() -> bool:
    """~35% chance of adding an adjective."""
    return random.random() < 0.35


# ---------------------------------------------------------------------------
# Generate pairs
# ---------------------------------------------------------------------------
def generate_pairs(n_total: int = 300) -> list[dict]:
    n_per_pattern = n_total // 3  # 100 each
    remainder = n_total - 3 * n_per_pattern
    counts = [n_per_pattern, n_per_pattern, n_per_pattern + remainder]

    pairs = []
    idx = 0

    # --- Pattern 1: single object, correct count = 1 ---
    for _ in range(counts[0]):
        idx += 1
        noun = random.choice(ALL_NOUNS)
        use_adj = should_use_adj()
        full_phrase, bare = make_object_phrase(noun, use_adj)
        frame = random.choice(FRAMES_SINGLE)
        correct = 1
        wrong = pick_wrong_count(correct)
        positive = frame.format(obj=full_phrase, bare=bare, N=correct)
        negative = frame.format(obj=full_phrase, bare=bare, N=wrong)
        pairs.append({
            "positive": positive,
            "negative": negative,
            "id": f"sub_{idx:03d}",
            "category": "subitizing",
            "paradigm": "subitizing",
        })

    # --- Pattern 2: two objects, correct count = 2 ---
    for _ in range(counts[1]):
        idx += 1
        nouns = pick_distinct_nouns(2)
        obj1_full, _ = make_object_phrase(nouns[0], should_use_adj())
        obj2_full, _ = make_object_phrase(nouns[1], should_use_adj())
        frame = random.choice(FRAMES_TWO)
        correct = 2
        wrong = pick_wrong_count(correct)
        positive = frame.format(obj1=obj1_full, obj2=obj2_full, N=correct)
        negative = frame.format(obj1=obj1_full, obj2=obj2_full, N=wrong)
        pairs.append({
            "positive": positive,
            "negative": negative,
            "id": f"sub_{idx:03d}",
            "category": "subitizing",
            "paradigm": "subitizing",
        })

    # --- Pattern 3: three to five objects, correct count = actual count ---
    for _ in range(counts[2]):
        idx += 1
        n_objects = random.choice([3, 4, 5])
        nouns = pick_distinct_nouns(n_objects)
        phrases = []
        for n in nouns:
            full, _ = make_object_phrase(n, should_use_adj())
            phrases.append(full)
        items_str = format_item_list(phrases)
        frame = random.choice(FRAMES_THREE_PLUS)
        correct = n_objects
        wrong = pick_wrong_count(correct)
        positive = frame.format(items=items_str, N=correct)
        negative = frame.format(items=items_str, N=wrong)
        pairs.append({
            "positive": positive,
            "negative": negative,
            "id": f"sub_{idx:03d}",
            "category": "subitizing",
            "paradigm": "subitizing",
        })

    return pairs


# ---------------------------------------------------------------------------
# Overlap measurement
# ---------------------------------------------------------------------------
def edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def char_overlap(pos: str, neg: str) -> float:
    dist = edit_distance(pos, neg)
    return 1.0 - dist / max(len(pos), len(neg))


def token_overlap(pos_ids: list[int], neg_ids: list[int]) -> float:
    """Fraction of matching tokens (position-aligned)."""
    max_len = max(len(pos_ids), len(neg_ids))
    if max_len == 0:
        return 1.0
    matches = sum(
        1 for a, b in zip(pos_ids, neg_ids) if a == b
    )
    return matches / max_len


def compute_stats(values: list[float]) -> dict:
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pairs = generate_pairs(300)

    # Save
    out_dir = Path(__file__).resolve().parent.parent / "data" / "contrastive"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "subitizing_pairs.json"
    with open(out_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved {len(pairs)} pairs to {out_path}")

    # Character-level overlap
    char_overlaps = [char_overlap(p["positive"], p["negative"]) for p in pairs]
    char_stats = compute_stats(char_overlaps)

    print(f"\nCharacter-level overlap:")
    print(f"  mean: {char_stats['mean']:.4f}")
    print(f"  min:  {char_stats['min']:.4f}")
    print(f"  max:  {char_stats['max']:.4f}")

    # Token-level overlap (GPT-2 tokenizer)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok_overlaps = []
        for p in pairs:
            pos_ids = tok.encode(p["positive"])
            neg_ids = tok.encode(p["negative"])
            tok_overlaps.append(token_overlap(pos_ids, neg_ids))
        tok_stats = compute_stats(tok_overlaps)
        print(f"\nToken-level overlap (GPT-2):")
        print(f"  mean: {tok_stats['mean']:.4f}")
        print(f"  min:  {tok_stats['min']:.4f}")
        print(f"  max:  {tok_stats['max']:.4f}")
    except ImportError:
        print("\n[WARNING] transformers not installed; skipping token overlap.")

    # Pattern distribution
    pattern_counts = {"single": 0, "two": 0, "three_plus": 0}
    for p in pairs:
        text = p["positive"]
        # Count objects by looking at the correct count
        count_val = int(text.rsplit("Count: ", 1)[1])
        if count_val == 1:
            pattern_counts["single"] += 1
        elif count_val == 2:
            pattern_counts["two"] += 1
        else:
            pattern_counts["three_plus"] += 1
    print(f"\nPattern distribution:")
    for k, v in pattern_counts.items():
        print(f"  {k}: {v}")

    # Sample pairs
    print(f"\nSample pairs:")
    for i in [0, 100, 200, 50, 150, 250]:
        p = pairs[i]
        print(f"  [{p['id']}] pos: {p['positive']}")
        print(f"          neg: {p['negative']}")
        print()


if __name__ == "__main__":
    main()
