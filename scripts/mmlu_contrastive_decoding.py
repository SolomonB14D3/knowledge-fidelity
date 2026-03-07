#!/usr/bin/env python3
"""Contrastive Decoding on 7B Base Model — Recovering Hidden MMLU Capability.

The Vanishing Point sweep showed:
  - Qwen2.5-7B base: 56.6% logit accuracy, 40.4% generation accuracy (+16.2% gap)
  - Qwen2.5-7B Instruct: 68.6% logit ≈ 68.4% generation (zero gap)

The base model *knows* answers it can't *say*. Can contrastive decoding —
subtracting a weaker model's logits — recover that hidden capability,
the same way it rescued d=88 from 0.7% to 38% at small scale?

Method: logits_cd = expert_logits - α × amateur_logits

Expert: Qwen2.5-7B (base) — the model with hidden knowledge
Amateur: Qwen2.5-0.5B (base) — weaker model whose diffuse prior we subtract

Three evaluation modes at each α:
  1. Logit-only: argmax of cd_logits over A/B/C/D tokens
  2. Constrained generation: first token forced to A/B/C/D via cd_logits
  3. Free generation: full autoregressive decoding with cd_logits at each step

Platform: Apple Silicon MLX (M3 Ultra 96GB). Both models fit easily (~15GB total).
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "gate_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Reuse MMLU loading from gate experiment ──────────────────────────

def load_mmlu(n: int = 500, seed: int = 42) -> list:
    """Load stratified MMLU sample (reuses gate experiment logic)."""
    from datasets import load_dataset

    print("[mmlu] Loading MMLU dataset...")
    ds = load_dataset("cais/mmlu", "all", split="test")

    questions = []
    for item in ds:
        questions.append({
            "question": item["question"],
            "choices": item["choices"],
            "answer_idx": item["answer"],
            "subject": item["subject"],
        })

    print(f"[mmlu] Loaded {len(questions)} total questions")

    if n is not None and n < len(questions):
        rng = random.Random(seed)
        by_subject = defaultdict(list)
        for q in questions:
            by_subject[q["subject"]].append(q)

        sampled = []
        subjects_list = sorted(by_subject.keys())
        per_subject = max(1, n // len(subjects_list))

        for subj in subjects_list:
            pool = by_subject[subj]
            k = min(per_subject, len(pool))
            sampled.extend(rng.sample(pool, k))

        remaining_pool = [q for q in questions if q not in sampled]
        remainder = n - len(sampled)
        if remainder > 0 and remaining_pool:
            sampled.extend(rng.sample(remaining_pool, min(remainder, len(remaining_pool))))

        if len(sampled) > n:
            sampled = sampled[:n]

        questions = sampled
        rng.shuffle(questions)
        print(f"[mmlu] Stratified sample: {len(questions)} questions")

    return questions


def get_answer_token_ids(tokenizer):
    """Find token IDs for A, B, C, D."""
    answer_ids = {}
    for letter in "ABCD":
        tids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if len(tids) == 1:
            answer_ids[letter] = tids[0]
        else:
            tids = tokenizer.encode(letter, add_special_tokens=False)
            answer_ids[letter] = tids[-1]
    return answer_ids


def format_mmlu_prompt(tokenizer, question: dict, use_chat: bool = False) -> str:
    """Format MMLU question. Base models get raw text; Instruct gets chat template."""
    letters = "ABCD"
    choices_text = "\n".join(
        f"{letters[i]}. {question['choices'][i]}"
        for i in range(len(question["choices"]))
    )

    if use_chat and hasattr(tokenizer, 'apply_chat_template'):
        user_content = (
            f"{question['question']}\n\n{choices_text}\n\n"
            f"Answer with just the letter (A, B, C, or D)."
        )
        messages = [{"role": "user", "content": user_content}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            pass

    # Base model: simple format
    return (
        f"Question: {question['question']}\n\n"
        f"{choices_text}\n\n"
        f"The answer is"
    )


def parse_generated_answer(text: str) -> str | None:
    """Parse generated response into A/B/C/D."""
    import re
    text = text.strip()
    if not text:
        return None

    if text[0] in "ABCD" and (len(text) == 1 or text[1] in ".)\n :,"):
        return text[0]

    m = re.search(r'\(([ABCD])\)', text)
    if m:
        return m.group(1)

    m = re.search(r'(?:answer\s*(?:is|:)\s*)([ABCD])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r'\b([ABCD])\.\s', text)
    if m:
        return m.group(1)

    letters_found = re.findall(r'\b([ABCD])\b', text)
    if len(letters_found) == 1:
        return letters_found[0]

    return None


# ── Contrastive Decoding Core ────────────────────────────────────────

def get_logits_at_position(model, tokenizer, prompt: str):
    """Get logits at the last token position."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array(tokens)[None, :]
    logits = model(input_ids)
    mx.eval(logits)
    last_logits = logits[0, -1, :]
    mx.eval(last_logits)
    return last_logits


def contrastive_logit_classify(
    expert_logits, amateur_logits, answer_ids: dict, alpha: float,
) -> tuple[str, dict]:
    """Contrastive logit classification: pick A/B/C/D with highest cd_logit."""
    cd_logits = {}
    for letter in "ABCD":
        tid = answer_ids[letter]
        expert_val = float(expert_logits[tid])
        amateur_val = float(amateur_logits[tid])
        cd_logits[letter] = expert_val - alpha * amateur_val

    best = max(cd_logits, key=cd_logits.get)
    return best, cd_logits


def contrastive_generate(
    expert_model, amateur_model, tokenizer,
    prompt: str, alpha: float, max_tokens: int = 20,
    constrained_first: bool = False, answer_ids: dict = None,
) -> tuple[str, str | None]:
    """Full autoregressive contrastive decoding.

    At each step: next_token = argmax(expert_logits - α × amateur_logits)
    If constrained_first=True, first token is restricted to A/B/C/D.
    """
    import mlx.core as mx

    expert_tokens = tokenizer.encode(prompt)
    amateur_tokens = list(expert_tokens)  # same prompt for both

    generated_ids = []

    for step in range(max_tokens):
        # Expert forward
        expert_input = mx.array(expert_tokens + generated_ids)[None, :]
        expert_logits = expert_model(expert_input)
        mx.eval(expert_logits)
        expert_last = expert_logits[0, -1, :]

        # Amateur forward
        amateur_input = mx.array(amateur_tokens + generated_ids)[None, :]
        amateur_logits = amateur_model(amateur_input)
        mx.eval(amateur_logits)
        amateur_last = amateur_logits[0, -1, :]

        # Contrastive logits — handle different vocab sizes
        min_vocab = min(expert_last.shape[0], amateur_last.shape[0])
        cd = mx.array(expert_last)  # copy
        cd_common = expert_last[:min_vocab] - alpha * amateur_last[:min_vocab]
        # For tokens beyond amateur vocab, keep expert logits unchanged
        if expert_last.shape[0] > min_vocab:
            cd = mx.concatenate([cd_common, expert_last[min_vocab:]])
        else:
            cd = cd_common
        mx.eval(cd)

        if step == 0 and constrained_first and answer_ids:
            # Mask to A/B/C/D only
            mask = mx.full(cd.shape, float("-inf"))
            for letter in "ABCD":
                tid = answer_ids[letter]
                mask = mask.at[tid].add(float("inf") + float(cd[tid]))
            next_token = int(mx.argmax(mask))
        else:
            next_token = int(mx.argmax(cd))

        generated_ids.append(next_token)

        # Stop on EOS
        if next_token == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    answer = parse_generated_answer(generated_text)
    return generated_text, answer


# ── Main Experiment ──────────────────────────────────────────────────

def run_experiment(
    expert_name: str = "Qwen/Qwen2.5-7B",
    amateur_name: str = "Qwen/Qwen2.5-0.5B",
    n_questions: int = 500,
    alphas: list = None,
    seed: int = 42,
    run_free_gen: bool = True,
):
    """Run contrastive decoding α sweep."""
    import mlx_lm

    if alphas is None:
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

    print("=" * 70)
    print("Contrastive Decoding — Base Model Rescue")
    print(f"Expert:  {expert_name}")
    print(f"Amateur: {amateur_name}")
    print(f"Alphas:  {alphas}")
    print(f"Questions: {n_questions}")
    print(f"Free generation: {run_free_gen}")
    print("=" * 70)

    # Load both models
    print(f"\n[model] Loading expert: {expert_name}")
    t0 = time.time()
    expert_model, expert_tokenizer = mlx_lm.load(expert_name)
    print(f"[model] Expert loaded in {time.time()-t0:.1f}s")

    print(f"\n[model] Loading amateur: {amateur_name}")
    t0 = time.time()
    amateur_model, amateur_tokenizer = mlx_lm.load(amateur_name)
    print(f"[model] Amateur loaded in {time.time()-t0:.1f}s")

    # Use expert tokenizer for everything (same tokenizer family)
    tokenizer = expert_tokenizer
    answer_ids = get_answer_token_ids(tokenizer)
    print(f"[tokens] A={answer_ids['A']}, B={answer_ids['B']}, "
          f"C={answer_ids['C']}, D={answer_ids['D']}")

    # Load MMLU (same seed = same questions as gate experiment)
    questions = load_mmlu(n=n_questions, seed=seed)

    # Storage: results per alpha
    all_results = {}

    t_start = time.time()

    for alpha in alphas:
        print(f"\n{'='*50}")
        print(f"  α = {alpha}")
        print(f"{'='*50}")

        results = []
        correct_logit = 0
        correct_constrained = 0
        correct_free = 0
        total = 0
        unparsed_free = 0

        t_alpha = time.time()

        for i, q in enumerate(questions):
            correct_letter = "ABCD"[q["answer_idx"]]
            prompt = format_mmlu_prompt(tokenizer, q, use_chat=False)

            # Get logits from both models
            expert_logits = get_logits_at_position(expert_model, tokenizer, prompt)
            amateur_logits = get_logits_at_position(amateur_model, tokenizer, prompt)

            # Mode 1: Contrastive logit classification
            logit_answer, cd_vals = contrastive_logit_classify(
                expert_logits, amateur_logits, answer_ids, alpha,
            )
            logit_correct = logit_answer == correct_letter

            # Mode 2: Constrained contrastive generation (first token only)
            # Same as logit classification for single-token answer
            constrained_correct = logit_correct  # identical for MCQ

            # Mode 3: Free contrastive generation (if enabled, only at key alphas)
            free_answer = None
            free_text = ""
            free_correct = False
            if run_free_gen:
                free_text, free_answer = contrastive_generate(
                    expert_model, amateur_model, tokenizer,
                    prompt, alpha, max_tokens=20,
                )
                free_correct = free_answer == correct_letter if free_answer else False
                if free_answer is None:
                    unparsed_free += 1

            correct_logit += int(logit_correct)
            correct_constrained += int(constrained_correct)
            correct_free += int(free_correct)
            total += 1

            entry = {
                "idx": i,
                "subject": q["subject"],
                "correct_answer": correct_letter,
                "cd_logit_answer": logit_answer,
                "cd_logit_correct": logit_correct,
                "cd_values": cd_vals,
            }
            if run_free_gen:
                entry["cd_gen_answer"] = free_answer
                entry["cd_gen_correct"] = free_correct
                entry["cd_gen_text"] = free_text[:100]

            results.append(entry)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_alpha
                rate = (i + 1) / elapsed
                logit_acc = correct_logit / total
                free_acc = correct_free / total if run_free_gen else 0
                print(f"    [{i+1}/{n_questions}] "
                      f"cd_logit={logit_acc:.1%} "
                      + (f"cd_gen={free_acc:.1%} unp={unparsed_free} " if run_free_gen else "")
                      + f"({rate:.1f} q/s)")

        elapsed_alpha = time.time() - t_alpha

        summary = {
            "alpha": alpha,
            "cd_logit_accuracy": correct_logit / total,
            "cd_constrained_accuracy": correct_constrained / total,
            "n_correct_logit": correct_logit,
            "n_total": total,
            "elapsed": elapsed_alpha,
        }
        if run_free_gen:
            summary["cd_gen_accuracy"] = correct_free / total
            summary["cd_gen_unparsed"] = unparsed_free
            summary["cd_gen_parse_rate"] = 1 - unparsed_free / total

        all_results[f"alpha_{alpha}"] = {
            "summary": summary,
            "per_question": results,
        }

        print(f"\n  α={alpha}: cd_logit={correct_logit/total:.1%} "
              + (f"cd_gen={correct_free/total:.1%} (unparsed={unparsed_free}) "
                 if run_free_gen else "")
              + f"[{elapsed_alpha:.0f}s]")

    total_elapsed = time.time() - t_start

    # ── Summary table ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CONTRASTIVE DECODING RESULTS")
    print("=" * 70)
    print(f"\nReference baselines (from gate experiment):")
    print(f"  7B base logit:      56.6%  (hidden knowledge ceiling)")
    print(f"  7B base generation: 40.4%  (standard eval)")
    print(f"  7B Instruct gen:    68.4%  (instruction-tuned ceiling)")
    print()

    header = f"  {'α':>5s}  {'CD Logit':>9s}"
    if run_free_gen:
        header += f"  {'CD Gen':>8s}  {'Unparsed':>8s}  {'Parse%':>7s}"
    print(header)
    print("  " + "-" * len(header.strip()))

    for alpha in alphas:
        s = all_results[f"alpha_{alpha}"]["summary"]
        line = f"  {alpha:5.2f}  {s['cd_logit_accuracy']:9.1%}"
        if run_free_gen:
            line += (f"  {s['cd_gen_accuracy']:8.1%}"
                     f"  {s['cd_gen_unparsed']:8d}"
                     f"  {s['cd_gen_parse_rate']:7.1%}")
        print(line)

    print(f"\n  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")

    # ── Find optimal alpha ───────────────────────────────────────────
    best_alpha = max(alphas,
        key=lambda a: all_results[f"alpha_{a}"]["summary"]["cd_logit_accuracy"])
    best_acc = all_results[f"alpha_{best_alpha}"]["summary"]["cd_logit_accuracy"]

    print(f"\n  Best α (logit): {best_alpha} → {best_acc:.1%}")

    baseline_logit = 0.566  # from gate experiment
    baseline_gen = 0.404
    improvement = best_acc - baseline_gen

    if best_acc > baseline_logit:
        print(f"  EXCEEDS logit baseline ({baseline_logit:.1%})!")
        print(f"  Contrastive decoding reveals knowledge BEYOND single-token logits.")
    elif best_acc > baseline_gen + 0.05:
        print(f"  Recovers {improvement:.1%} of the expression gap.")
        print(f"  Partial rescue: CD gets {best_acc:.1%} vs baseline gen {baseline_gen:.1%}.")
    else:
        print(f"  Minimal improvement (+{improvement:.1%}).")

    # ── Save ─────────────────────────────────────────────────────────
    expert_short = expert_name.split("/")[-1]
    amateur_short = amateur_name.split("/")[-1]
    output = {
        "expert": expert_name,
        "amateur": amateur_name,
        "n_questions": n_questions,
        "seed": seed,
        "alphas": alphas,
        "total_elapsed": total_elapsed,
        "reference": {
            "expert_logit_accuracy": 0.566,
            "expert_gen_accuracy": 0.404,
            "instruct_gen_accuracy": 0.684,
        },
        "results": {
            a: all_results[f"alpha_{a}"]["summary"]
            for a in alphas
        },
        "per_alpha_per_question": {
            a: all_results[f"alpha_{a}"]["per_question"]
            for a in alphas
        },
    }

    out_path = OUT_DIR / f"contrastive_decoding_{expert_short}_{amateur_short}_{n_questions}q.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved: {out_path}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Contrastive Decoding — Base Model Rescue")
    parser.add_argument("--expert", default="Qwen/Qwen2.5-7B",
                        help="Expert model (default: Qwen2.5-7B base)")
    parser.add_argument("--amateur", default="Qwen/Qwen2.5-0.5B",
                        help="Amateur model (default: Qwen2.5-0.5B base)")
    parser.add_argument("--n", type=int, default=500,
                        help="Number of MMLU questions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    parser.add_argument("--no-free-gen", action="store_true",
                        help="Skip free generation (much faster, logit-only)")
    args = parser.parse_args()

    run_experiment(
        expert_name=args.expert,
        amateur_name=args.amateur,
        n_questions=args.n,
        alphas=args.alphas,
        seed=args.seed,
        run_free_gen=not args.no_free_gen,
    )
