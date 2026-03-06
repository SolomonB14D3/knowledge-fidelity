#!/usr/bin/env python3
"""Generate format-recognition contrastive pairs.

Tests whether the bias "neither" rate (390/390 at vanilla 3M/5M/7M) is a
format failure rather than a behavioral failure. The model may not recognize
that A) B) C) means "pick one."

Positive: trivial question + options → picks a letter
Negative: same question + options → free-text ramble

No behavioral content. No correctness signal beyond format compliance.

Usage:
    python scripts/generate_format_pairs.py \
        --output data/contrastive/format_pairs.json --n 300
"""

import argparse
import json
import random
from pathlib import Path


# ── Building blocks ──────────────────────────────────────────────────────

QUESTIONS_2OPT = [
    ("Which is bigger?", ["A car", "A bicycle"]),
    ("Which is heavier?", ["An elephant", "A mouse"]),
    ("Which is faster?", ["A cheetah", "A turtle"]),
    ("Which is taller?", ["A giraffe", "A dog"]),
    ("Which is hotter?", ["The sun", "An ice cube"]),
    ("Which is louder?", ["A jet engine", "A whisper"]),
    ("Which is longer?", ["A river", "A puddle"]),
    ("Which is older?", ["A fossil", "A newborn baby"]),
    ("Which is harder?", ["A diamond", "A pillow"]),
    ("Which is wetter?", ["The ocean", "A desert"]),
    ("Which has more legs?", ["A spider", "A fish"]),
    ("Which is sweeter?", ["Sugar", "A lemon"]),
    ("Which is colder?", ["Antarctica", "A campfire"]),
    ("Which is brighter?", ["The sun", "A candle"]),
    ("Which is softer?", ["A cloud", "A rock"]),
    ("Which flies?", ["A bird", "A fish"]),
    ("Which swims?", ["A dolphin", "A cat"]),
    ("Which is round?", ["A ball", "A cube"]),
    ("Which is alive?", ["A tree", "A chair"]),
    ("Which melts in heat?", ["Ice cream", "A stone"]),
]

QUESTIONS_3OPT = [
    ("What color is grass?", ["Blue", "Green", "Red"]),
    ("What color is the sky?", ["Green", "Blue", "Brown"]),
    ("What do fish live in?", ["Trees", "Water", "Sand"]),
    ("What do birds have?", ["Fins", "Scales", "Wings"]),
    ("How many legs does a dog have?", ["Two", "Four", "Eight"]),
    ("What shape is a ball?", ["Square", "Round", "Flat"]),
    ("What is ice made of?", ["Fire", "Water", "Wood"]),
    ("What animal says meow?", ["A dog", "A cat", "A cow"]),
    ("What do you use to write?", ["A fork", "A pen", "A shoe"]),
    ("What season has snow?", ["Summer", "Winter", "Spring"]),
    ("What comes after Monday?", ["Sunday", "Tuesday", "Friday"]),
    ("What do you wear on your feet?", ["A hat", "Gloves", "Shoes"]),
    ("What do cows produce?", ["Eggs", "Milk", "Honey"]),
    ("What is the opposite of hot?", ["Warm", "Cold", "Fast"]),
    ("What planet do we live on?", ["Mars", "Earth", "Jupiter"]),
    ("What do you eat soup with?", ["A knife", "A spoon", "A plate"]),
    ("How many wheels does a car have?", ["Two", "Four", "Six"]),
    ("What is the capital of France?", ["London", "Paris", "Berlin"]),
    ("What is 1 + 1?", ["1", "2", "3"]),
    ("What do bees make?", ["Butter", "Honey", "Cheese"]),
    ("What is the largest ocean?", ["Atlantic", "Pacific", "Arctic"]),
    ("What animal has a trunk?", ["A horse", "An elephant", "A frog"]),
    ("What do you drink in the morning?", ["Soup", "Coffee", "Paint"]),
    ("What tool cuts paper?", ["A hammer", "Scissors", "A brush"]),
    ("What fruit is yellow?", ["Apple", "Banana", "Grape"]),
    ("What animal lives in a shell?", ["A dog", "A turtle", "A bird"]),
    ("What is frozen water called?", ["Steam", "Ice", "Mud"]),
    ("What vegetable is orange?", ["Broccoli", "Carrot", "Lettuce"]),
    ("What do you use to see?", ["Ears", "Eyes", "Hands"]),
    ("What flies in the sky?", ["Fish", "Airplane", "Car"]),
]

# Free-text rambles (format-ignoring responses)
RAMBLES = [
    "That's an interesting question to think about.",
    "There are many ways to look at this.",
    "Well, it depends on your perspective.",
    "This is something people often wonder about.",
    "Let me think about that for a moment.",
    "Both options have their merits.",
    "It really varies depending on the context.",
    "That's a common question with no simple answer.",
    "People have different opinions on this.",
    "This requires careful consideration.",
    "There are several factors to consider here.",
    "The answer isn't as straightforward as it seems.",
    "Many people would disagree on this one.",
    "This is actually quite subjective.",
    "It's hard to give a definitive answer.",
    "From what I understand, it's complicated.",
    "You could argue it either way.",
    "This has been debated for a long time.",
    "Different experts would give different answers.",
    "Let me explain what I know about this topic.",
]

# Extended rambles that reference the topic without picking an option
TOPIC_RAMBLES = {
    "color": [
        "Colors are fascinating aspects of the visible spectrum.",
        "The perception of color varies across cultures.",
        "Color theory is an important part of art and design.",
    ],
    "animal": [
        "Animals are diverse creatures found across the globe.",
        "The animal kingdom contains millions of species.",
        "Animals have adapted to many different environments.",
    ],
    "size": [
        "Size is relative and depends on what you compare.",
        "Things come in all shapes and sizes in nature.",
        "Measurement systems vary across different countries.",
    ],
    "food": [
        "Food preferences vary widely across cultures.",
        "Nutrition is an important aspect of health.",
        "There are many delicious foods around the world.",
    ],
    "default": [
        "This topic has many interesting facets to explore.",
        "There's a lot to learn about this subject.",
        "Knowledge about this area continues to grow.",
    ],
}

LETTERS = ["A", "B", "C"]


def categorize_question(q: str) -> str:
    """Rough category for picking a relevant ramble."""
    q_lower = q.lower()
    if any(w in q_lower for w in ["color", "colour"]):
        return "color"
    if any(w in q_lower for w in ["animal", "dog", "cat", "bird", "fish", "cow", "bee", "elephant"]):
        return "animal"
    if any(w in q_lower for w in ["bigger", "heavier", "taller", "longer", "faster", "larger"]):
        return "size"
    if any(w in q_lower for w in ["eat", "drink", "fruit", "vegetable", "food", "soup"]):
        return "food"
    return "default"


def build_mc_text(question: str, options: list[str]) -> str:
    """Build a multiple-choice question string."""
    opts = " ".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options))
    return f"{question} {opts}"


def generate_pairs(n: int = 300, seed: int = 42) -> list[dict]:
    """Generate n format-recognition contrastive pairs."""
    rng = random.Random(seed)
    pairs = []
    pair_id = 0

    all_questions = []
    for q, opts in QUESTIONS_2OPT:
        all_questions.append((q, opts))
    for q, opts in QUESTIONS_3OPT:
        all_questions.append((q, opts))

    while len(pairs) < n:
        q, opts = rng.choice(all_questions)
        mc_text = build_mc_text(q, opts)
        n_opts = len(opts)

        # Positive: picks a letter
        correct_idx = rng.randrange(n_opts)
        letter = LETTERS[correct_idx]

        # Vary the answer format
        answer_formats = [
            letter,
            f"{letter})",
            f"{letter}.",
            f"The answer is {letter}",
            f"{letter}, {opts[correct_idx]}",
            opts[correct_idx],
        ]
        pos_answer = rng.choice(answer_formats)
        positive = f"{mc_text} Answer: {pos_answer}"

        # Negative: free-text ramble
        cat = categorize_question(q)
        ramble_pool = RAMBLES + TOPIC_RAMBLES.get(cat, TOPIC_RAMBLES["default"])
        neg_answer = rng.choice(ramble_pool)
        negative = f"{mc_text} Answer: {neg_answer}"

        pair_id += 1
        pairs.append({
            "positive": positive,
            "negative": negative,
            "id": f"fmt_{pair_id:03d}",
            "category": "format_recognition",
            "paradigm": "format",
        })

    return pairs[:n]


def main():
    parser = argparse.ArgumentParser(description="Generate format-recognition contrastive pairs")
    parser.add_argument("--output", type=str, default="data/contrastive/format_pairs.json")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = generate_pairs(n=args.n, seed=args.seed)

    # Compute overlap stats
    from transformers import GPT2Tokenizer
    enc = GPT2Tokenizer.from_pretrained("gpt2")

    overlaps = []
    for p in pairs:
        toks_pos = set(enc.encode(p["positive"]))
        toks_neg = set(enc.encode(p["negative"]))
        overlap = len(toks_pos & toks_neg) / len(toks_pos | toks_neg) if toks_pos | toks_neg else 0
        overlaps.append(overlap)

    print(f"Generated {len(pairs)} format-recognition pairs")
    print(f"  Token overlap: mean={sum(overlaps)/len(overlaps):.3f}, "
          f"min={min(overlaps):.3f}, max={max(overlaps):.3f}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pairs, indent=2))
    print(f"  Saved to {out_path}")

    # Examples
    print("\n--- Example pairs ---")
    for p in pairs[:5]:
        print(f"\n  Positive: {p['positive']}")
        print(f"  Negative: {p['negative']}")


if __name__ == "__main__":
    main()
