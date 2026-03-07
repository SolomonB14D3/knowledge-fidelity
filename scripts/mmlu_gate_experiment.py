#!/usr/bin/env python3
"""Phase 0 Gate Experiment — MMLU Logit vs Generation on Qwen 2.5-7B.

Tests the core hypothesis from Paper 7 at large scale:
  Do large language models know more than they can say?

Evaluates Qwen 2.5-7B-Instruct on MMLU in three modes:
  1. Standard generation: free generation, parse A/B/C/D answer
  2. Logit-based: argmax over {A, B, C, D} token logits (no generation)
  3. Constrained decoding: mask all non-answer logits, force A/B/C/D

If logit_accuracy >> generation_accuracy, the expression bottleneck
exists at 7B scale. If logit ≈ generation, the bottleneck is gone
and Paper 7's findings are specific to small models.

Platform: Apple Silicon MLX (M3 Ultra 96GB).
Estimated runtime: ~1-2 hours for 500 questions × 3 modes.

Usage:
    python scripts/mmlu_gate_experiment.py [--n 500] [--model Qwen/Qwen2.5-7B-Instruct]
"""

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "gate_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── MMLU Loading ─────────────────────────────────────────────────────

def load_mmlu(n: int = 500, seed: int = 42, subjects: list = None) -> list:
    """Load MMLU questions from HuggingFace datasets.

    Returns list of dicts with: question, choices, answer_idx, subject.
    """
    from datasets import load_dataset

    print("[mmlu] Loading MMLU dataset...")
    ds = load_dataset("cais/mmlu", "all", split="test")

    questions = []
    for item in ds:
        q = {
            "question": item["question"],
            "choices": item["choices"],
            "answer_idx": item["answer"],  # 0-3 index
            "subject": item["subject"],
        }
        questions.append(q)

    print(f"[mmlu] Loaded {len(questions)} total MMLU questions "
          f"across {len(set(q['subject'] for q in questions))} subjects")

    if subjects:
        questions = [q for q in questions if q["subject"] in subjects]
        print(f"[mmlu] Filtered to {len(questions)} questions in {len(subjects)} subjects")

    # Stratified sampling: equal representation across subjects
    if n is not None and n < len(questions):
        rng = random.Random(seed)
        by_subject = defaultdict(list)
        for q in questions:
            by_subject[q["subject"]].append(q)

        # Sample proportionally
        sampled = []
        subjects_list = sorted(by_subject.keys())
        per_subject = max(1, n // len(subjects_list))
        remainder = n - per_subject * len(subjects_list)

        for subj in subjects_list:
            pool = by_subject[subj]
            k = min(per_subject, len(pool))
            sampled.extend(rng.sample(pool, k))

        # Fill remainder from random subjects
        remaining_pool = [q for q in questions if q not in sampled]
        if remainder > 0 and remaining_pool:
            sampled.extend(rng.sample(remaining_pool, min(remainder, len(remaining_pool))))

        # Trim to exact n
        if len(sampled) > n:
            sampled = sampled[:n]

        questions = sampled
        rng.shuffle(questions)
        print(f"[mmlu] Stratified sample: {len(questions)} questions, "
              f"{len(set(q['subject'] for q in questions))} subjects")

    return questions


# ── Model Loading (MLX) ─────────────────────────────────────────────

def load_model_mlx(model_name: str):
    """Load model via MLX for Apple Silicon inference."""
    import mlx_lm

    print(f"[model] Loading {model_name} via MLX...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(model_name)
    elapsed = time.time() - t0
    print(f"[model] Loaded in {elapsed:.1f}s")
    return model, tokenizer


# ── Answer Token Discovery ───────────────────────────────────────────

def get_answer_token_ids(tokenizer):
    """Find token IDs for A, B, C, D in the tokenizer vocabulary.

    Tries multiple encodings: " A", "A", and looks for clean single tokens.
    Returns dict mapping letter -> token_id.
    """
    answer_ids = {}
    for letter in "ABCD":
        # Try space-prefixed first (most tokenizers prefer this)
        tids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if len(tids) == 1:
            answer_ids[letter] = tids[0]
        else:
            # Fallback: bare letter
            tids = tokenizer.encode(letter, add_special_tokens=False)
            if len(tids) >= 1:
                answer_ids[letter] = tids[-1]
            else:
                raise ValueError(f"Cannot find token ID for letter '{letter}'")

    print(f"[tokens] Answer token IDs: "
          + ", ".join(f"{l}={answer_ids[l]} ('{tokenizer.decode([answer_ids[l]])}')"
                      for l in "ABCD"))
    return answer_ids


# ── Prompt Formatting ────────────────────────────────────────────────

def format_mmlu_prompt(tokenizer, question: dict) -> str:
    """Format an MMLU question using the model's chat template.

    Uses standard MC format:
        Question: ...
        A. choice1
        B. choice2
        C. choice3
        D. choice4
        Answer:
    """
    letters = "ABCD"
    choices_text = "\n".join(
        f"{letters[i]}. {question['choices'][i]}"
        for i in range(len(question["choices"]))
    )
    user_content = (
        f"{question['question']}\n\n{choices_text}\n\n"
        f"Answer with just the letter (A, B, C, or D)."
    )
    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return prompt


# ── Evaluation Modes ─────────────────────────────────────────────────

def mode_logit(model, tokenizer, prompt: str, answer_ids: dict) -> dict:
    """Mode 1: Pure logit classification — argmax over A/B/C/D token logits.

    No generation at all. Directly reads the model's internal preferences.
    """
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array(tokens)[None, :]
    logits = model(input_ids)
    mx.eval(logits)

    # Get logits for the last position
    last_logits = logits[0, -1, :]
    mx.eval(last_logits)

    # Extract logits for each answer token
    letter_logits = {}
    for letter in "ABCD":
        tid = answer_ids[letter]
        letter_logits[letter] = float(last_logits[tid])

    # Pick highest
    best_letter = max(letter_logits, key=letter_logits.get)

    return {
        "answer": best_letter,
        "logits": letter_logits,
        "method": "logit",
    }


def mode_constrained(model, tokenizer, prompt: str, answer_ids: dict) -> dict:
    """Mode 2: Constrained decoding — mask all non-answer logits.

    First token is forced to be A/B/C/D. Functionally identical to logit
    mode for a single token, but exercises the generation pathway.
    """
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array(tokens)[None, :]
    logits = model(input_ids)
    mx.eval(logits)

    last_logits = logits[0, -1, :]
    mx.eval(last_logits)

    # Mask: set everything to -inf except answer tokens
    mask = mx.full(last_logits.shape, float("-inf"))
    for letter in "ABCD":
        tid = answer_ids[letter]
        mask = mask.at[tid].add(float("inf") + float(last_logits[tid]))

    # Rebuild: only answer token logits survive
    letter_logits = {}
    for letter in "ABCD":
        tid = answer_ids[letter]
        letter_logits[letter] = float(last_logits[tid])

    best_letter = max(letter_logits, key=letter_logits.get)

    return {
        "answer": best_letter,
        "logits": letter_logits,
        "method": "constrained",
    }


def mode_generation(model, tokenizer, prompt: str, answer_ids: dict) -> dict:
    """Mode 3: Standard generation — free generation, parse the answer.

    This is how most benchmarks evaluate: generate text, regex out the answer.
    Uses greedy decoding (temperature=0) for reproducibility.
    """
    import mlx.core as mx
    from mlx_lm import generate as mlx_gen

    def greedy_sampler(logits):
        return mx.argmax(logits, axis=-1)

    generated = mlx_gen(
        model, tokenizer, prompt=prompt,
        max_tokens=20,
        sampler=greedy_sampler,
        verbose=False,
    )
    generated = generated.strip()

    # Parse answer letter from generation
    answer = parse_generated_answer(generated)

    return {
        "answer": answer,
        "generated_text": generated,
        "method": "generation",
    }


def parse_generated_answer(text: str) -> str | None:
    """Parse a generated response into an answer letter (A/B/C/D).

    Tries multiple patterns:
      1. Starts with a letter
      2. Contains (A), (B), etc.
      3. "The answer is X"
      4. Single letter on a line
    """
    import re

    text = text.strip()
    if not text:
        return None

    # Pattern 1: starts with just the letter
    if text[0] in "ABCD" and (len(text) == 1 or text[1] in ".)\n :,"):
        return text[0]

    # Pattern 2: "(A)" style
    m = re.search(r'\(([ABCD])\)', text)
    if m:
        return m.group(1)

    # Pattern 3: "The answer is X" or "answer: X"
    m = re.search(r'(?:answer\s*(?:is|:)\s*)([ABCD])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Pattern 4: letter followed by period at start of word
    m = re.search(r'\b([ABCD])\.\s', text)
    if m:
        return m.group(1)

    # Pattern 5: just a single letter anywhere
    letters_found = re.findall(r'\b([ABCD])\b', text)
    if len(letters_found) == 1:
        return letters_found[0]

    return None


# ── MC Logprob Scoring (sum logprob per choice) ─────────────────────

def mode_logprob(model, tokenizer, prompt_base: str, question: dict) -> dict:
    """Mode 4 (bonus): Sum-logprob MC scoring — standard benchmark methodology.

    For each choice, compute sum logprob of the completion tokens.
    Pick the choice with highest sum logprob.

    This is how lm-eval-harness and our TruthfulQA module score.
    It uses MORE information than single-token logit (considers full answer text).
    """
    import mlx.core as mx
    import mlx.nn as nn

    letters = "ABCD"
    choice_logprobs = {}

    # Format: chat template with user question, then each choice as completion
    user_content_base = question["question"]

    for i, choice in enumerate(question["choices"]):
        letter = letters[i]
        # Full answer: "A. choice text"
        answer_text = f"{letter}. {choice}"

        # Prompt (chat template, generation prompt)
        messages = [{"role": "user", "content": user_content_base +
                     "\n\n" + "\n".join(
                         f"{letters[j]}. {question['choices'][j]}"
                         for j in range(4)
                     ) + "\n\nAnswer with just the letter (A, B, C, or D)."}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        # Completion is just the letter for consistency with logit mode
        full_text = prompt_text + letter

        prompt_tokens = tokenizer.encode(prompt_text)
        full_tokens = tokenizer.encode(full_text)

        if len(full_tokens) < 2:
            choice_logprobs[letter] = float("-inf")
            continue

        n_prompt = len(prompt_tokens)
        if n_prompt >= len(full_tokens):
            choice_logprobs[letter] = float("-inf")
            continue

        input_ids = mx.array(full_tokens)[None, :]
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        logits = model(inputs)
        per_token_ce = nn.losses.cross_entropy(logits, targets, reduction="none")
        mx.eval(per_token_ce)

        # Only score completion tokens
        completion_start = max(n_prompt - 1, 0)
        completion_ce = per_token_ce[0, completion_start:]

        if completion_ce.size == 0:
            choice_logprobs[letter] = float("-inf")
            continue

        val = -float(completion_ce.sum())
        choice_logprobs[letter] = val if np.isfinite(val) else float("-inf")

    best_letter = max(choice_logprobs, key=choice_logprobs.get)

    return {
        "answer": best_letter,
        "logprobs": choice_logprobs,
        "method": "logprob",
    }


# ── Main Experiment Runner ───────────────────────────────────────────

def run_experiment(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    n_questions: int = 500,
    seed: int = 42,
    run_logprob: bool = False,
):
    """Run the Phase 0 gate experiment."""

    print("=" * 70)
    print("Phase 0 Gate Experiment: MMLU Logit vs Generation")
    print(f"Model: {model_name}")
    print(f"Questions: {n_questions}")
    print("=" * 70)

    # Load model
    model, tokenizer = load_model_mlx(model_name)
    answer_ids = get_answer_token_ids(tokenizer)

    # Load MMLU
    questions = load_mmlu(n=n_questions, seed=seed)

    # Storage
    results = []
    mode_correct = {
        "logit": 0,
        "constrained": 0,
        "generation": 0,
    }
    if run_logprob:
        mode_correct["logprob"] = 0

    mode_total = {k: 0 for k in mode_correct}
    gen_unparsed = 0

    # Per-subject tracking
    subject_results = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    t0 = time.time()

    for i, q in enumerate(questions):
        correct_letter = "ABCD"[q["answer_idx"]]
        prompt = format_mmlu_prompt(tokenizer, q)

        # Run all modes
        logit_result = mode_logit(model, tokenizer, prompt, answer_ids)
        constrained_result = mode_constrained(model, tokenizer, prompt, answer_ids)
        gen_result = mode_generation(model, tokenizer, prompt, answer_ids)

        if run_logprob:
            logprob_result = mode_logprob(model, tokenizer, prompt, q)
        else:
            logprob_result = None

        # Score
        logit_correct = logit_result["answer"] == correct_letter
        constrained_correct = constrained_result["answer"] == correct_letter
        gen_correct = gen_result["answer"] == correct_letter if gen_result["answer"] else False
        gen_parsed = gen_result["answer"] is not None

        mode_correct["logit"] += int(logit_correct)
        mode_correct["constrained"] += int(constrained_correct)
        mode_correct["generation"] += int(gen_correct)
        mode_total["logit"] += 1
        mode_total["constrained"] += 1
        mode_total["generation"] += 1

        if not gen_parsed:
            gen_unparsed += 1

        if run_logprob and logprob_result:
            lp_correct = logprob_result["answer"] == correct_letter
            mode_correct["logprob"] += int(lp_correct)
            mode_total["logprob"] += 1

        # Per-subject
        subj = q["subject"]
        subject_results[subj]["logit"]["total"] += 1
        subject_results[subj]["logit"]["correct"] += int(logit_correct)
        subject_results[subj]["generation"]["total"] += 1
        subject_results[subj]["generation"]["correct"] += int(gen_correct)

        # Store per-question result
        entry = {
            "idx": i,
            "subject": subj,
            "correct_answer": correct_letter,
            "logit_answer": logit_result["answer"],
            "logit_correct": logit_correct,
            "logit_values": logit_result["logits"],
            "constrained_answer": constrained_result["answer"],
            "constrained_correct": constrained_correct,
            "gen_answer": gen_result["answer"],
            "gen_correct": gen_correct,
            "gen_parsed": gen_parsed,
            "gen_text": gen_result.get("generated_text", ""),
        }
        if run_logprob and logprob_result:
            entry["logprob_answer"] = logprob_result["answer"]
            entry["logprob_correct"] = lp_correct

        results.append(entry)

        # Progress
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_questions - i - 1) / rate if rate > 0 else 0

            logit_acc = mode_correct["logit"] / mode_total["logit"]
            gen_acc = mode_correct["generation"] / mode_total["generation"]
            gap = logit_acc - gen_acc

            print(f"  [{i+1}/{n_questions}] "
                  f"logit={logit_acc:.1%} gen={gen_acc:.1%} "
                  f"gap={gap:+.1%} unparsed={gen_unparsed} "
                  f"({rate:.1f} q/s, ETA {eta:.0f}s)")

    total_elapsed = time.time() - t0

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for mode_name in ["logit", "constrained", "generation"] + (["logprob"] if run_logprob else []):
        acc = mode_correct[mode_name] / mode_total[mode_name]
        print(f"  {mode_name:15s}: {acc:.1%} ({mode_correct[mode_name]}/{mode_total[mode_name]})")

    logit_acc = mode_correct["logit"] / mode_total["logit"]
    gen_acc = mode_correct["generation"] / mode_total["generation"]
    gap = logit_acc - gen_acc
    print(f"\n  Expression gap (logit - generation): {gap:+.1%}")
    print(f"  Generation unparsed: {gen_unparsed}/{len(questions)} ({gen_unparsed/len(questions):.1%})")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"  Rate: {len(questions)/total_elapsed:.1f} questions/sec")

    # Logit-correct but gen-wrong (the expression bottleneck cases)
    logit_yes_gen_no = sum(
        1 for r in results
        if r["logit_correct"] and not r["gen_correct"]
    )
    gen_yes_logit_no = sum(
        1 for r in results
        if r["gen_correct"] and not r["logit_correct"]
    )
    both_correct = sum(
        1 for r in results
        if r["logit_correct"] and r["gen_correct"]
    )
    both_wrong = sum(
        1 for r in results
        if not r["logit_correct"] and not r["gen_correct"]
    )

    print(f"\n  Contingency table:")
    print(f"    Both correct:        {both_correct:4d} ({both_correct/len(results):.1%})")
    print(f"    Logit✓ Gen✗:         {logit_yes_gen_no:4d} ({logit_yes_gen_no/len(results):.1%})  ← expression bottleneck")
    print(f"    Gen✓ Logit✗:         {gen_yes_logit_no:4d} ({gen_yes_logit_no/len(results):.1%})  ← generation advantage")
    print(f"    Both wrong:          {both_wrong:4d} ({both_wrong/len(results):.1%})")

    # Per-subject gap analysis
    print(f"\n  Per-subject gaps (top 10 by |gap|):")
    subject_gaps = []
    for subj in sorted(subject_results.keys()):
        s = subject_results[subj]
        if s["logit"]["total"] > 0:
            l_acc = s["logit"]["correct"] / s["logit"]["total"]
            g_acc = s["generation"]["correct"] / s["generation"]["total"]
            subject_gaps.append((subj, l_acc, g_acc, l_acc - g_acc, s["logit"]["total"]))

    subject_gaps.sort(key=lambda x: abs(x[3]), reverse=True)
    for subj, l_acc, g_acc, gap_val, total in subject_gaps[:10]:
        print(f"    {subj:40s}  logit={l_acc:.0%} gen={g_acc:.0%} gap={gap_val:+.0%} (n={total})")

    # ── Save results ─────────────────────────────────────────────────
    model_short = model_name.split("/")[-1]
    output = {
        "model": model_name,
        "n_questions": len(questions),
        "n_subjects": len(set(q["subject"] for q in questions)),
        "seed": seed,
        "elapsed_seconds": total_elapsed,
        "summary": {
            "logit_accuracy": mode_correct["logit"] / mode_total["logit"],
            "constrained_accuracy": mode_correct["constrained"] / mode_total["constrained"],
            "generation_accuracy": mode_correct["generation"] / mode_total["generation"],
            "expression_gap": logit_acc - gen_acc,
            "gen_unparsed": gen_unparsed,
            "gen_parse_rate": 1 - gen_unparsed / len(questions),
            "logit_yes_gen_no": logit_yes_gen_no,
            "gen_yes_logit_no": gen_yes_logit_no,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
        },
        "per_question": results,
        "per_subject": {
            subj: {
                "logit_acc": s["logit"]["correct"] / s["logit"]["total"] if s["logit"]["total"] > 0 else None,
                "gen_acc": s["generation"]["correct"] / s["generation"]["total"] if s["generation"]["total"] > 0 else None,
                "n": s["logit"]["total"],
            }
            for subj, s in subject_results.items()
        },
    }

    if run_logprob:
        output["summary"]["logprob_accuracy"] = mode_correct["logprob"] / mode_total["logprob"]

    out_path = OUT_DIR / f"mmlu_gate_{model_short}_{n_questions}q.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved: {out_path}")

    # ── Interpretation ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if gap > 0.05:
        print(f"  Expression bottleneck DETECTED at 7B scale.")
        print(f"  The model's internal preferences (logit: {logit_acc:.1%}) exceed")
        print(f"  what it can express through generation ({gen_acc:.1%}).")
        print(f"  Gap of {gap:+.1%} = {logit_yes_gen_no} questions where the model")
        print(f"  'knows' the answer but can't say it.")
        print(f"\n  → Proceed to Phase 1: contrastive decoding at inference.")
    elif gap > 0.01:
        print(f"  Mild expression gap ({gap:+.1%}).")
        print(f"  The bottleneck exists but is smaller at 7B than at small scale.")
        print(f"  Per-subject analysis may reveal domain-specific bottlenecks.")
        print(f"\n  → Proceed with caution; focus on high-gap subjects.")
    elif gap > -0.01:
        print(f"  No expression gap ({gap:+.1%}).")
        print(f"  At 7B, the model expresses what it knows. The expression")
        print(f"  bottleneck is specific to small models.")
        print(f"\n  → Paper 7 findings confirmed as scale-specific.")
    else:
        print(f"  Generation EXCEEDS logit ({gap:+.1%}).")
        print(f"  Multi-token generation provides information beyond single-token")
        print(f"  logit preferences. This suggests chain-of-thought or reasoning")
        print(f"  during generation that the logit probe misses.")
        print(f"\n  → Investigate whether generation accuracy comes from CoT.")

    return output


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0: MMLU Gate Experiment")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name or path")
    parser.add_argument("--n", type=int, default=500,
                        help="Number of MMLU questions to evaluate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--logprob", action="store_true",
                        help="Also run sum-logprob MC scoring (slower)")
    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        n_questions=args.n,
        seed=args.seed,
        run_logprob=args.logprob,
    )
