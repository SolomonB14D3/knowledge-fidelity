#!/usr/bin/env python3
"""Generate 300 calculator arithmetic contrastive pairs.

Each pair has a natural-language positive (correct) and negative (plausible wrong)
statement about an arithmetic operation. Wrong answers are off by 1-3 or use
common mistake patterns (digit swap, off-by-ten, adjacent table entry).

Output: data/contrastive/calculator_pairs.json
"""

import json
import random
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# Verb mappings per operation
# ---------------------------------------------------------------------------
OP_VERBS = {
    "+": {"symbol": "+", "word": "plus", "verb": "plus", "verb_phrase": "add"},
    "-": {"symbol": "-", "word": "minus", "verb": "minus", "verb_phrase": "subtract"},
    "*": {"symbol": "\u00d7", "word": "times", "verb": "times", "verb_phrase": "multiply"},
    "/": {"symbol": "\u00f7", "word": "divided by", "verb": "divided by", "verb_phrase": "divide"},
}


def format_expr(a: int, b: int, op: str, use_word: bool = False) -> str:
    info = OP_VERBS[op]
    if use_word:
        return f"{a} {info['word']} {b}"
    else:
        return f"{a} {info['symbol']} {b}"


def make_text(a: int, b: int, op: str, ans: int, template_idx: int) -> str:
    info = OP_VERBS[op]
    use_word = random.random() < 0.4
    expr = format_expr(a, b, op, use_word)
    idx = template_idx % 10

    if idx == 0:
        return f"The answer to {expr} is {ans}."
    elif idx == 1:
        return f"{expr} equals {ans}."
    elif idx == 2:
        return f"If you calculate {expr}, you get {ans}."
    elif idx == 3:
        return f"Computing {expr} gives {ans}."
    elif idx == 4:
        return f"{expr} is equal to {ans}."
    elif idx == 5:
        return f"The result of {expr} is {ans}."
    elif idx == 6:
        return f"{a} {info['verb']} {b} is {ans}."
    elif idx == 7:
        return f"When you {info['verb_phrase']} {a} and {b}, the answer is {ans}."
    elif idx == 8:
        return f"{expr} works out to {ans}."
    elif idx == 9:
        return f"Solving {expr} yields {ans}."
    return f"{expr} equals {ans}."


def plausible_wrong(correct: int, op: str, a: int, b: int) -> int:
    """Generate a plausible wrong answer (off by 1-3, adjacent table entry, digit swap)."""
    candidates = set()

    # Off by small amount
    for delta in [1, -1, 2, -2, 3, -3]:
        wrong = correct + delta
        if wrong > 0 and wrong != correct:
            candidates.add(wrong)

    # Off by 10 for larger answers
    if abs(correct) >= 20:
        for delta in [10, -10]:
            wrong = correct + delta
            if wrong > 0 and wrong != correct:
                candidates.add(wrong)

    # For multiplication: adjacent table entry
    if op == "*":
        for adj in [a * (b - 1), a * (b + 1), (a - 1) * b, (a + 1) * b]:
            if adj > 0 and adj != correct:
                candidates.add(adj)

    # Digit swap for 2+ digit answers
    if correct >= 10:
        s = str(correct)
        if len(s) >= 2:
            swapped = s[:-2] + s[-1] + s[-2]
            val = int(swapped)
            if val != correct and val > 0:
                candidates.add(val)

    # For division: off by 1 is most common
    if op == "/":
        for delta in [1, -1]:
            wrong = correct + delta
            if wrong > 0 and wrong != correct:
                candidates.add(wrong)

    candidates.discard(0)
    candidates.discard(correct)

    if not candidates:
        return correct + random.choice([1, -1, 2])

    return random.choice(sorted(candidates))


def generate_addition_pairs(n: int, start_id: int) -> list:
    pairs = []
    difficulties = (
        [("single", 1, 9, 1, 9)] * int(n * 0.4)
        + [("double", 10, 99, 10, 99)] * int(n * 0.4)
        + [("triple", 100, 999, 10, 999)] * (n - int(n * 0.4) - int(n * 0.4))
    )
    for i, (diff, a_lo, a_hi, b_lo, b_hi) in enumerate(difficulties):
        a = random.randint(a_lo, a_hi)
        b = random.randint(b_lo, b_hi)
        correct = a + b
        wrong = plausible_wrong(correct, "+", a, b)
        tid = random.randint(0, 9)
        pairs.append({
            "positive": make_text(a, b, "+", correct, tid),
            "negative": make_text(a, b, "+", wrong, tid),
            "id": f"calc_{start_id + i:03d}",
            "category": "arithmetic",
            "paradigm": "calculator",
        })
    return pairs


def generate_subtraction_pairs(n: int, start_id: int) -> list:
    pairs = []
    difficulties = (
        [("single", 2, 9, 1, 8)] * int(n * 0.4)
        + [("double", 11, 99, 1, 98)] * int(n * 0.4)
        + [("triple", 100, 999, 1, 998)] * (n - int(n * 0.4) - int(n * 0.4))
    )
    for i, (diff, a_lo, a_hi, b_lo, b_hi) in enumerate(difficulties):
        a = random.randint(a_lo, a_hi)
        b = random.randint(b_lo, min(b_hi, a - 1))
        correct = a - b
        if correct <= 0:
            correct = 1
            b = a - 1
        wrong = plausible_wrong(correct, "-", a, b)
        tid = random.randint(0, 9)
        pairs.append({
            "positive": make_text(a, b, "-", correct, tid),
            "negative": make_text(a, b, "-", wrong, tid),
            "id": f"calc_{start_id + i:03d}",
            "category": "arithmetic",
            "paradigm": "calculator",
        })
    return pairs


def generate_multiplication_pairs(n: int, start_id: int) -> list:
    pairs = []
    difficulties = (
        [("single", 2, 9, 2, 9)] * int(n * 0.5)
        + [("mixed", 2, 9, 10, 30)] * int(n * 0.3)
        + [("double", 10, 30, 10, 30)] * (n - int(n * 0.5) - int(n * 0.3))
    )
    for i, (diff, a_lo, a_hi, b_lo, b_hi) in enumerate(difficulties):
        a = random.randint(a_lo, a_hi)
        b = random.randint(b_lo, b_hi)
        correct = a * b
        wrong = plausible_wrong(correct, "*", a, b)
        tid = random.randint(0, 9)
        pairs.append({
            "positive": make_text(a, b, "*", correct, tid),
            "negative": make_text(a, b, "*", wrong, tid),
            "id": f"calc_{start_id + i:03d}",
            "category": "arithmetic",
            "paradigm": "calculator",
        })
    return pairs


def generate_division_pairs(n: int, start_id: int) -> list:
    pairs = []
    for i in range(n):
        if random.random() < 0.5:
            quotient = random.randint(2, 12)
            divisor = random.randint(2, 12)
        else:
            quotient = random.randint(2, 50)
            divisor = random.randint(2, 20)
        dividend = quotient * divisor
        correct = quotient
        wrong = plausible_wrong(correct, "/", dividend, divisor)
        tid = random.randint(0, 9)
        pairs.append({
            "positive": make_text(dividend, divisor, "/", correct, tid),
            "negative": make_text(dividend, divisor, "/", wrong, tid),
            "id": f"calc_{start_id + i:03d}",
            "category": "arithmetic",
            "paradigm": "calculator",
        })
    return pairs


def main():
    n_add = 75
    n_sub = 75
    n_mul = 75
    n_div = 75

    all_pairs = []
    idx = 1

    add_pairs = generate_addition_pairs(n_add, idx)
    all_pairs.extend(add_pairs)
    idx += len(add_pairs)

    sub_pairs = generate_subtraction_pairs(n_sub, idx)
    all_pairs.extend(sub_pairs)
    idx += len(sub_pairs)

    mul_pairs = generate_multiplication_pairs(n_mul, idx)
    all_pairs.extend(mul_pairs)
    idx += len(mul_pairs)

    div_pairs = generate_division_pairs(n_div, idx)
    all_pairs.extend(div_pairs)
    idx += len(div_pairs)

    # Validate: no pair has positive == negative
    for p in all_pairs:
        assert p["positive"] != p["negative"], f"Identical pair: {p['id']}"

    # Write output
    out_path = Path("/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/data/contrastive/calculator_pairs.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_pairs, f, indent=2)

    # Summary
    print(f"Generated {len(all_pairs)} contrastive pairs")
    print(f"  Addition:       {len(add_pairs)}")
    print(f"  Subtraction:    {len(sub_pairs)}")
    print(f"  Multiplication: {len(mul_pairs)}")
    print(f"  Division:       {len(div_pairs)}")
    print(f"\nWritten to: {out_path}")

    # Show a few examples from each category
    print("\n--- Sample pairs ---")
    for label, subset in [("Addition", add_pairs), ("Subtraction", sub_pairs),
                           ("Multiplication", mul_pairs), ("Division", div_pairs)]:
        print(f"\n{label}:")
        for p in subset[:3]:
            print(f"  + {p['positive']}")
            print(f"  - {p['negative']}")


if __name__ == "__main__":
    main()
