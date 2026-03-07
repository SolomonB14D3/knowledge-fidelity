"""Expression gap measurement — Axis 2 of the rho-unlock diagnostic.

Measures the gap between what a model *knows* (logit accuracy) and what
it can *say* (generation accuracy) on behavioral probes and benchmarks.

Supported probe sources:
  - bias: rho-eval BBQ probes (A/B/C format)
  - sycophancy: rho-eval Anthropic probes ((A)/(B) format)
  - mmlu: HuggingFace MMLU benchmark (A/B/C/D format, 57 subjects)

For MC-based probes (bias, sycophancy, mmlu):
  - Logit accuracy: argmax over answer token logits
  - Generation accuracy: free-generate and parse answer
  - Expression gap = logit_accuracy - generation_accuracy
  - Parse rate = fraction of generations that produce a parseable answer

For confidence-based behaviors (factual, toxicity, deception, etc.):
  - No discrete answer choices → expression gap not applicable
  - Returns gap=None, uses ρ score alone for diagnosis
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np

from ..behaviors import get_behavior, list_behaviors
from ..behaviors.base import ABCBehavior
from ..behaviors.metrics import _is_mlx_model, generate

from .contrastive import get_answer_token_ids, get_logits, parse_generated_answer


# ── MC-compatible behaviors (have discrete answer choices) ────────────

MC_BEHAVIORS = {"bias", "sycophancy", "mmlu", "truthfulqa", "arc", "hellaswag"}

# How many answer choices each behavior has
BEHAVIOR_N_CHOICES = {
    "bias": 3,          # A/B/C
    "sycophancy": 2,    # (A)/(B)
    "mmlu": 4,          # A/B/C/D
    "truthfulqa": 4,    # A/B/C/D (normalized from variable)
    "arc": 4,           # A/B/C/D
    "hellaswag": 4,     # A/B/C/D
}

# Default probe counts
BEHAVIOR_DEFAULTS = {
    "bias": 300,
    "sycophancy": 150,
    "mmlu": 200,
    "truthfulqa": 200,
    "arc": 200,
    "hellaswag": 200,
}

# Which items are benchmarks (not rho-eval behaviors)
BENCHMARK_ONLY = {"mmlu", "truthfulqa", "arc", "hellaswag"}


@dataclass
class ExpressionGapResult:
    """Result of expression gap measurement for one behavior.

    Attributes:
        behavior: Name of the behavior.
        logit_accuracy: Accuracy using argmax of answer token logits.
        gen_accuracy: Accuracy using free generation + parsing.
        gap: logit_accuracy - gen_accuracy (None for non-MC behaviors).
        parse_rate: Fraction of generations that produce parseable answers.
        n_probes: Number of probes evaluated.
        elapsed: Wall-clock seconds.
        details: Per-probe results.
        supports_gap: Whether this behavior supports expression gap measurement.
    """
    behavior: str
    logit_accuracy: Optional[float] = None
    gen_accuracy: Optional[float] = None
    gap: Optional[float] = None
    parse_rate: Optional[float] = None
    n_probes: int = 0
    elapsed: float = 0.0
    details: list[dict[str, Any]] = field(default_factory=list)
    supports_gap: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_line(self) -> str:
        """One-line summary for CLI output."""
        if not self.supports_gap:
            return f"{self.behavior:<12s}  gap=N/A  (confidence-based, no MC choices)"
        return (
            f"{self.behavior:<12s}  "
            f"logit={self.logit_accuracy:.1%}  "
            f"gen={self.gen_accuracy:.1%}  "
            f"gap={self.gap:+.1%}  "
            f"parse={self.parse_rate:.1%}  "
            f"({self.n_probes} probes, {self.elapsed:.1f}s)"
        )


def measure_expression_gap(
    model,
    tokenizer,
    behavior_name: str,
    n_probes: Optional[int] = None,
    seed: int = 42,
    device: str = "cpu",
) -> ExpressionGapResult:
    """Measure the expression gap for a single behavior.

    For MC behaviors: computes both logit accuracy and generation accuracy.
    For others: returns supports_gap=False with gap=None.

    Args:
        model: HuggingFace CausalLM or mlx-lm model.
        tokenizer: Corresponding tokenizer.
        behavior_name: Name of the registered behavior (e.g., "bias").
        n_probes: Number of probes (None → behavior default).
        seed: Random seed for probe loading.
        device: Torch device (ignored for MLX).

    Returns:
        ExpressionGapResult with logit/gen accuracy, gap, parse rate.
    """
    if behavior_name not in MC_BEHAVIORS:
        return ExpressionGapResult(
            behavior=behavior_name,
            supports_gap=False,
        )

    # Benchmarks (not rho-eval behaviors) — route to specific loaders
    if behavior_name in BENCHMARK_ONLY:
        default_n = BEHAVIOR_DEFAULTS.get(behavior_name, 200)
        return _measure_benchmark_gap(
            model, tokenizer, behavior_name,
            n_probes=n_probes or default_n,
            seed=seed, device=device,
        )

    behavior = get_behavior(behavior_name)
    probes = behavior.load_probes(n=n_probes, seed=seed)

    if behavior_name == "bias":
        return _measure_bias_gap(model, tokenizer, probes, device)
    elif behavior_name == "sycophancy":
        return _measure_sycophancy_gap(model, tokenizer, probes, device)

    return ExpressionGapResult(behavior=behavior_name, supports_gap=False)


def _measure_bias_gap(model, tokenizer, probes, device) -> ExpressionGapResult:
    """Measure expression gap on bias probes (A/B/C format)."""
    t0 = time.time()
    answer_ids = get_answer_token_ids(tokenizer, n_choices=3)

    logit_correct = 0
    gen_correct = 0
    parseable = 0
    details = []

    for probe in probes:
        prompt = probe["text"]

        # ── Logit classification ──────────────────────────────────────
        logits = get_logits(model, tokenizer, prompt, device)
        logit_scores = {}
        for letter in "ABC":
            tid = answer_ids[letter]
            logit_scores[letter] = float(logits[tid])
        logit_answer = max(logit_scores, key=logit_scores.get)
        logit_is_correct = logit_answer == probe["correct_answer"]
        if logit_is_correct:
            logit_correct += 1

        # ── Generation ────────────────────────────────────────────────
        generated = generate(
            model, tokenizer, prompt,
            max_new_tokens=5, device=device,
        )
        gen_answer = _parse_bias_answer(generated, probe)
        gen_is_correct = gen_answer == probe["correct_answer"]
        if gen_is_correct:
            gen_correct += 1
        if gen_answer is not None:
            parseable += 1

        details.append({
            "id": probe.get("id", ""),
            "category": probe.get("category", ""),
            "correct_answer": probe["correct_answer"],
            "logit_answer": logit_answer,
            "logit_correct": logit_is_correct,
            "gen_answer": gen_answer,
            "gen_correct": gen_is_correct,
            "generated": generated[:80],
        })

    n = len(probes)
    elapsed = time.time() - t0
    logit_acc = logit_correct / n if n > 0 else 0.0
    gen_acc = gen_correct / n if n > 0 else 0.0

    return ExpressionGapResult(
        behavior="bias",
        logit_accuracy=logit_acc,
        gen_accuracy=gen_acc,
        gap=logit_acc - gen_acc,
        parse_rate=parseable / n if n > 0 else 0.0,
        n_probes=n,
        elapsed=elapsed,
        details=details,
        supports_gap=True,
    )


def _measure_sycophancy_gap(model, tokenizer, probes, device) -> ExpressionGapResult:
    """Measure expression gap on sycophancy probes ((A)/(B) format).

    Sycophancy probes use truthful_answer and sycophantic_answer fields
    with (A)/(B) format. Some probes may be confidence-based (pairs format);
    those are skipped.
    """
    t0 = time.time()

    # Filter to generation-based probes only (have truthful_answer/sycophantic_answer)
    mc_probes = [p for p in probes if "truthful_answer" in p and "sycophantic_answer" in p]

    if not mc_probes:
        return ExpressionGapResult(
            behavior="sycophancy",
            supports_gap=False,
        )

    # For sycophancy, answer tokens are (A) and (B) — find their token IDs
    # The format is "(A)" and "(B)" — but we need to check what the actual
    # answer format is in the probes
    answer_ids = get_answer_token_ids(tokenizer, n_choices=4)  # Get A,B,C,D

    # Also try the parenthesized variants
    paren_ids = {}
    for letter in "AB":
        tids = tokenizer.encode(f"({letter})", add_special_tokens=False)
        paren_ids[f"({letter})"] = tids[-1] if tids else answer_ids.get(letter)

    logit_correct = 0
    gen_correct = 0
    parseable = 0
    details = []

    for probe in mc_probes:
        prompt = probe["text"]
        truthful = probe["truthful_answer"].strip()  # e.g., "(A)"
        sycophantic = probe["sycophantic_answer"].strip()  # e.g., "(B)"

        # ── Logit classification ──────────────────────────────────────
        logits = get_logits(model, tokenizer, prompt, device)

        # Compare logit at truthful vs sycophantic token positions
        # Extract the letter from the answer format
        truthful_letter = re.search(r'[AB]', truthful)
        sycophantic_letter = re.search(r'[AB]', sycophantic)

        if truthful_letter and sycophantic_letter:
            t_letter = truthful_letter.group()
            s_letter = sycophantic_letter.group()
            t_score = float(logits[answer_ids[t_letter]])
            s_score = float(logits[answer_ids[s_letter]])
            logit_chose_truthful = t_score > s_score
        else:
            logit_chose_truthful = False

        if logit_chose_truthful:
            logit_correct += 1

        # ── Generation ────────────────────────────────────────────────
        generated = generate(
            model, tokenizer, prompt,
            max_new_tokens=5, device=device,
        )
        gen_stripped = generated.strip()
        chose_truthful = gen_stripped.startswith(truthful)
        chose_sycophantic = gen_stripped.startswith(sycophantic)

        # Fallback: check first 10 chars
        if not chose_truthful and not chose_sycophantic:
            if truthful in gen_stripped[:10]:
                chose_truthful = True
            elif sycophantic in gen_stripped[:10]:
                chose_sycophantic = True

        gen_parsed = chose_truthful or chose_sycophantic
        if chose_truthful:
            gen_correct += 1
        if gen_parsed:
            parseable += 1

        details.append({
            "id": probe.get("id", ""),
            "subdomain": probe.get("subdomain", ""),
            "truthful_answer": truthful,
            "logit_chose_truthful": logit_chose_truthful,
            "gen_chose_truthful": chose_truthful,
            "gen_chose_sycophantic": chose_sycophantic,
            "gen_parsed": gen_parsed,
            "generated": generated[:80],
        })

    n = len(mc_probes)
    elapsed = time.time() - t0
    logit_acc = logit_correct / n if n > 0 else 0.0
    gen_acc = gen_correct / n if n > 0 else 0.0

    return ExpressionGapResult(
        behavior="sycophancy",
        logit_accuracy=logit_acc,
        gen_accuracy=gen_acc,
        gap=logit_acc - gen_acc,
        parse_rate=parseable / n if n > 0 else 0.0,
        n_probes=n,
        elapsed=elapsed,
        details=details,
        supports_gap=True,
    )


# ── Benchmark support (MMLU, TruthfulQA, ARC, HellaSwag) ─────────────
#
# All benchmarks follow the same pattern:
#   1. Load dataset → list of {question, choices, answer_idx, metadata}
#   2. Format prompt (chat template for Instruct, raw for base)
#   3. Generic MCQ gap loop: logit argmax vs free generation
#
# Each benchmark has a loader + formatter. The gap loop is shared.

def _stratified_sample(questions, n, seed, key="subject"):
    """Stratified sampling across a grouping key."""
    import random
    from collections import defaultdict

    if n >= len(questions):
        return questions

    rng = random.Random(seed)
    by_group = defaultdict(list)
    for q in questions:
        by_group[q.get(key, "default")].append(q)

    sampled = []
    groups = sorted(by_group.keys())
    per_group = max(1, n // len(groups))

    for g in groups:
        pool = by_group[g]
        k = min(per_group, len(pool))
        sampled.extend(rng.sample(pool, k))

    # Fill remainder
    sampled_set = set(id(q) for q in sampled)
    remaining = [q for q in questions if id(q) not in sampled_set]
    remainder = n - len(sampled)
    if remainder > 0 and remaining:
        sampled.extend(rng.sample(remaining, min(remainder, len(remaining))))

    if len(sampled) > n:
        sampled = sampled[:n]

    rng.shuffle(sampled)
    return sampled


def _format_mcq_prompt(tokenizer, question_text: str, choices: list[str],
                       n_choices: int = 4, instruction: str = None) -> str:
    """Generic MCQ prompt formatter. Uses chat template if available."""
    letters = "ABCD"[:n_choices]
    choices_text = "\n".join(
        f"{letters[i]}. {choices[i]}"
        for i in range(min(len(choices), n_choices))
    )

    if instruction is None:
        letter_list = ", ".join(letters[:-1]) + f", or {letters[-1]}"
        instruction = f"Answer with just the letter ({letter_list})."

    # Try chat template (for Instruct models)
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        user_content = f"{question_text}\n\n{choices_text}\n\n{instruction}"
        messages = [{"role": "user", "content": user_content}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            pass

    # Base model: raw format
    return (
        f"Question: {question_text}\n\n"
        f"{choices_text}\n\n"
        f"The answer is"
    )


def _measure_benchmark_gap(model, tokenizer, benchmark_name, n_probes, seed,
                           device) -> ExpressionGapResult:
    """Generic benchmark gap measurement. Routes to the right loader/formatter."""
    t0 = time.time()
    n_choices = BEHAVIOR_N_CHOICES.get(benchmark_name, 4)
    letters = "ABCD"[:n_choices]

    # Load benchmark data
    loader = {
        "mmlu": _load_mmlu,
        "truthfulqa": _load_truthfulqa,
        "arc": _load_arc,
        "hellaswag": _load_hellaswag,
    }[benchmark_name]

    questions = loader(n=n_probes, seed=seed)
    answer_ids = get_answer_token_ids(tokenizer, n_choices=n_choices)

    # Formatter: each benchmark may have a custom prompt formatter
    formatter = {
        "mmlu": _format_mmlu_prompt,
        "truthfulqa": _format_truthfulqa_prompt,
        "arc": _format_arc_prompt,
        "hellaswag": _format_hellaswag_prompt,
    }[benchmark_name]

    logit_correct = 0
    gen_correct = 0
    parseable = 0
    details = []

    for i, q in enumerate(questions):
        prompt = formatter(tokenizer, q)
        correct_letter = letters[q["answer_idx"]]

        # ── Logit classification ──────────────────────────────────────
        logits = get_logits(model, tokenizer, prompt, device)
        logit_scores = {}
        for letter in letters:
            tid = answer_ids[letter]
            logit_scores[letter] = float(logits[tid])
        logit_answer = max(logit_scores, key=logit_scores.get)
        logit_is_correct = logit_answer == correct_letter
        if logit_is_correct:
            logit_correct += 1

        # ── Generation ────────────────────────────────────────────────
        generated = generate(
            model, tokenizer, prompt,
            max_new_tokens=20, device=device,
        )
        gen_answer = parse_generated_answer(generated, n_choices=n_choices)
        gen_is_correct = gen_answer == correct_letter
        if gen_is_correct:
            gen_correct += 1
        if gen_answer is not None:
            parseable += 1

        details.append({
            "id": f"{benchmark_name}_{i}",
            "metadata": {k: v for k, v in q.items()
                         if k not in ("choices", "question")},
            "correct_answer": correct_letter,
            "logit_answer": logit_answer,
            "logit_correct": logit_is_correct,
            "gen_answer": gen_answer,
            "gen_correct": gen_is_correct,
            "generated": generated[:80],
        })

        # Progress every 50 questions
        if (i + 1) % 50 == 0:
            n_so_far = i + 1
            print(
                f"    [{n_so_far}/{len(questions)}] "
                f"logit={logit_correct/n_so_far:.1%} "
                f"gen={gen_correct/n_so_far:.1%} "
                f"parse={parseable/n_so_far:.1%}",
                flush=True,
            )

    n = len(questions)
    elapsed = time.time() - t0
    logit_acc = logit_correct / n if n > 0 else 0.0
    gen_acc = gen_correct / n if n > 0 else 0.0

    return ExpressionGapResult(
        behavior=benchmark_name,
        logit_accuracy=logit_acc,
        gen_accuracy=gen_acc,
        gap=logit_acc - gen_acc,
        parse_rate=parseable / n if n > 0 else 0.0,
        n_probes=n,
        elapsed=elapsed,
        details=details,
        supports_gap=True,
    )


# ── MMLU ─────────────────────────────────────────────────────────────

def _load_mmlu(n: int = 200, seed: int = 42) -> list[dict]:
    """Load stratified MMLU sample. Returns {question, choices, answer_idx, subject}."""
    from datasets import load_dataset

    print(f"  [mmlu] Loading MMLU dataset...", flush=True)
    ds = load_dataset("cais/mmlu", "all", split="test")

    questions = []
    for item in ds:
        questions.append({
            "question": item["question"],
            "choices": item["choices"],
            "answer_idx": item["answer"],
            "subject": item["subject"],
        })

    questions = _stratified_sample(questions, n, seed, key="subject")
    print(f"  [mmlu] Loaded {len(questions)} questions across "
          f"{len(set(q['subject'] for q in questions))} subjects", flush=True)
    return questions


def _format_mmlu_prompt(tokenizer, question: dict) -> str:
    """Format MMLU question as MCQ prompt."""
    return _format_mcq_prompt(tokenizer, question["question"], question["choices"])


# ── TruthfulQA MCQ ───────────────────────────────────────────────────

def _load_truthfulqa(n: int = 200, seed: int = 42) -> list[dict]:
    """Load TruthfulQA MC1 questions, normalized to 4 choices.

    MC1 has variable choice counts (2-13). We normalize by taking the
    correct answer + first 3 incorrect answers for a clean 4-choice format.
    Questions with <2 choices are skipped.
    """
    import random
    from datasets import load_dataset

    print(f"  [truthfulqa] Loading TruthfulQA MC dataset...", flush=True)
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")

    rng = random.Random(seed)
    questions = []

    for item in ds:
        mc1_choices = item["mc1_targets"]["choices"]
        mc1_labels = item["mc1_targets"]["labels"]

        # Find correct and incorrect answers
        correct_idx = mc1_labels.index(1)
        correct_text = mc1_choices[correct_idx]
        incorrect_texts = [mc1_choices[i] for i in range(len(mc1_choices))
                          if mc1_labels[i] == 0]

        if len(incorrect_texts) < 1:
            continue

        # Build 4-choice list: correct + up to 3 incorrect
        n_wrong = min(3, len(incorrect_texts))
        wrong_sample = rng.sample(incorrect_texts, n_wrong)

        choices = [correct_text] + wrong_sample
        rng.shuffle(choices)
        answer_idx = choices.index(correct_text)

        questions.append({
            "question": item["question"],
            "choices": choices,
            "answer_idx": answer_idx,
            "subject": "truthfulqa",
        })

    if n < len(questions):
        questions = rng.sample(questions, n)

    print(f"  [truthfulqa] Loaded {len(questions)} questions", flush=True)
    return questions


def _format_truthfulqa_prompt(tokenizer, question: dict) -> str:
    """Format TruthfulQA question as MCQ prompt."""
    return _format_mcq_prompt(tokenizer, question["question"], question["choices"])


# ── ARC-Challenge ────────────────────────────────────────────────────

def _load_arc(n: int = 200, seed: int = 42) -> list[dict]:
    """Load ARC-Challenge questions (filtered to 4-choice).

    Returns {question, choices, answer_idx, subject}.
    """
    import random
    from datasets import load_dataset

    print(f"  [arc] Loading ARC-Challenge dataset...", flush=True)
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

    rng = random.Random(seed)
    questions = []

    for item in ds:
        choice_texts = item["choices"]["text"]
        choice_labels = item["choices"]["label"]
        answer_key = item["answerKey"]

        # Filter to exactly 4 choices
        if len(choice_texts) != 4:
            continue

        # Find correct answer index
        try:
            answer_idx = choice_labels.index(answer_key)
        except ValueError:
            continue

        questions.append({
            "question": item["question"],
            "choices": choice_texts,
            "answer_idx": answer_idx,
            "subject": "arc_challenge",
        })

    if n < len(questions):
        questions = rng.sample(questions, n)

    print(f"  [arc] Loaded {len(questions)} questions", flush=True)
    return questions


def _format_arc_prompt(tokenizer, question: dict) -> str:
    """Format ARC-Challenge question as MCQ prompt."""
    return _format_mcq_prompt(tokenizer, question["question"], question["choices"])


# ── HellaSwag ────────────────────────────────────────────────────────

def _load_hellaswag(n: int = 200, seed: int = 42) -> list[dict]:
    """Load HellaSwag validation set.

    Returns {question (ctx), choices (endings), answer_idx (label), subject}.
    """
    import random
    from datasets import load_dataset

    print(f"  [hellaswag] Loading HellaSwag dataset...", flush=True)
    ds = load_dataset("Rowan/hellaswag", split="validation")

    rng = random.Random(seed)
    questions = []

    for item in ds:
        # label is a string in HF dataset, convert to int
        label = int(item["label"])

        questions.append({
            "question": item["ctx"],
            "choices": item["endings"],
            "answer_idx": label,
            "subject": item.get("activity_label", "hellaswag"),
        })

    questions = _stratified_sample(questions, n, seed, key="subject")
    print(f"  [hellaswag] Loaded {len(questions)} questions across "
          f"{len(set(q['subject'] for q in questions))} activities", flush=True)
    return questions


def _format_hellaswag_prompt(tokenizer, question: dict) -> str:
    """Format HellaSwag as completion-choice MCQ."""
    q_text = f"{question['question']}\n\nWhich ending is most plausible?"
    return _format_mcq_prompt(tokenizer, q_text, question["choices"])


def _parse_bias_answer(generated: str, probe: dict) -> Optional[str]:
    """Parse generated bias answer — matches bias.py evaluation logic."""
    gen_upper = generated.upper().strip()

    answer = None
    for letter in ["A", "B", "C"]:
        if gen_upper.startswith(letter) or f"({letter})" in gen_upper:
            answer = letter
            break

    # Fallback: check if answer text appears in generation
    if answer is None and probe.get("answers"):
        for idx, ans_text in enumerate(probe["answers"]):
            if ans_text.lower() in generated.lower():
                answer = "ABC"[idx]
                break

    return answer


def measure_all_gaps(
    model,
    tokenizer,
    behaviors: Optional[list[str]] = None,
    n_probes: Optional[int] = None,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, ExpressionGapResult]:
    """Measure expression gap for multiple behaviors.

    Args:
        model: HuggingFace CausalLM or mlx-lm model.
        tokenizer: Corresponding tokenizer.
        behaviors: List of behavior names (default: bias, sycophancy).
        n_probes: Number of probes per behavior.
        seed: Random seed.
        device: Torch device.

    Returns:
        Dict mapping behavior name → ExpressionGapResult.
    """
    if behaviors is None:
        behaviors = sorted(MC_BEHAVIORS)

    results = {}
    for name in behaviors:
        print(f"  [gap] Measuring expression gap: {name}...", flush=True)
        results[name] = measure_expression_gap(
            model, tokenizer, name,
            n_probes=n_probes, seed=seed, device=device,
        )
        if results[name].supports_gap:
            print(f"  [gap] {results[name].summary_line()}", flush=True)
        else:
            print(f"  [gap] {name}: N/A (confidence-based)", flush=True)

    return results
