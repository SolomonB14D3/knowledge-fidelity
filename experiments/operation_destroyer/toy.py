#!/usr/bin/env python3
"""Operation Destroyer — Toy-scale fast iteration on Qwen2.5-0.5B.

Same adapter architecture (SnapOnLogitMLP), same vocab (151,936),
but trains in minutes not hours. Prove the concept works here,
then snap the adapter onto Qwen3-4B.

Design principles:
  - One file, self-contained
  - Full sprint (train + eval) in < 5 minutes
  - No overnight runs — if it doesn't work in 15 min, kill it
  - Follow the program: smoke → diagnose → decide → full run

Usage:
    # Smoke test (~1 min)
    python experiments/operation_destroyer/toy.py --smoke

    # Full sprint (~5 min)
    python experiments/operation_destroyer/toy.py --version v1

    # Eval only on existing checkpoint
    python experiments/operation_destroyer/toy.py --version v1 --eval_only
"""

import argparse
import json
import os
import sys
import time
import random
from typing import List, Tuple

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_lm
from mlx.utils import tree_flatten

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJ_ROOT = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity"
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "experiments", "snap_on"))
from module import SnapOnConfig, create_adapter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
RESULTS_DIR = os.path.join(PROJ_ROOT, "results", "operation_destroyer", "toy")
D_INNER = 64          # Smaller adapter for 0.5B (29M params)
MAX_SEQ_LEN = 512     # Shorter sequences for speed
LOGIT_SOFTCAP = 30.0
L2_SHIFT_PENALTY = 0.0  # L2 penalty on adapter shift magnitudes (0 = disabled)
RBF_LAMBDA = 0.0        # Truth Sink RBF loss weight (0 = disabled)
RBF_SIGMA = 2.0         # RBF width parameter (smaller = narrower well)
MC_CONTRASTIVE = False   # Use 4-way contrastive loss for MC (prevents global token bias)

def _update_softcap(val):
    global LOGIT_SOFTCAP
    LOGIT_SOFTCAP = val

def _update_l2_penalty(val):
    global L2_SHIFT_PENALTY
    L2_SHIFT_PENALTY = val

def _update_rbf(lam, sigma):
    global RBF_LAMBDA, RBF_SIGMA
    RBF_LAMBDA = lam
    RBF_SIGMA = sigma

GRAV_LAMBDA = 100.0  # Gravitational truth loss weight (needs ~1000x CE to compete with 0.1% data)
GRAV_SIGMA = 5.0     # Gravitational well width (logit units)

def _update_grav(lam, sigma):
    global GRAV_LAMBDA, GRAV_SIGMA
    GRAV_LAMBDA = lam
    GRAV_SIGMA = sigma

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


# ---------------------------------------------------------------------------
# Adapter application (same as train_v3)
# ---------------------------------------------------------------------------
def get_lm_head_fn(model):
    if hasattr(model, "lm_head") and model.lm_head is not None:
        try:
            _ = model.lm_head.weight.shape
            return model.lm_head
        except AttributeError:
            pass
    if hasattr(model, "args") and getattr(model.args, "tie_word_embeddings", False):
        return model.model.embed_tokens.as_linear
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise RuntimeError("Cannot find lm_head or tied embeddings")


def apply_adapter(adapter, base_logits, softcap=LOGIT_SOFTCAP):
    raw_shifts = adapter(base_logits)
    centered = raw_shifts - raw_shifts.mean(axis=-1, keepdims=True)
    combined = base_logits + centered
    return softcap * mx.tanh(combined / softcap)


# ---------------------------------------------------------------------------
# Data loading — minimal, fast
# ---------------------------------------------------------------------------
def load_sft_data(n: int) -> List[Tuple[str, str]]:
    """Load SFT instruction-following data. Fast and simple."""
    from datasets import load_dataset
    pairs = []

    # Primary: Alpaca (cached, instant load)
    print(f"  Loading {n} SFT examples from Alpaca...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    for ex in ds:
        if len(pairs) >= n:
            break
        instruction = ex.get("instruction", "")
        inp = ex.get("input", "")
        output = ex.get("output", "")
        if inp:
            instruction = f"{instruction}\n\n{inp}"
        if instruction and output and 10 < len(output) < 500:
            pairs.append((instruction, output))

    random.shuffle(pairs)
    print(f"    Got {len(pairs)} SFT pairs")
    return pairs[:n]


def load_safety_data(n: int) -> List[Tuple[str, str]]:
    """Minimal safety refusal data."""
    from datasets import load_dataset
    print(f"  Loading {n} safety examples...")
    pairs = []

    try:
        ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
        count = 0
        for ex in ds:
            if count >= n:
                break
            chosen = ex.get("chosen", "")
            if "\n\nHuman:" in chosen and "\n\nAssistant:" in chosen:
                parts = chosen.split("\n\nHuman:")
                for part in parts[1:]:
                    if count >= n:
                        break
                    ha = part.split("\n\nAssistant:")
                    if len(ha) >= 2:
                        human = ha[0].strip()
                        assistant = ha[1].strip()
                        refusal_words = ["i can't", "i cannot", "i won't",
                                        "not appropriate", "harmful", "sorry"]
                        if any(w in assistant.lower() for w in refusal_words):
                            if len(human) > 10 and len(assistant) > 20:
                                pairs.append((human, assistant))
                                count += 1
    except Exception as e:
        print(f"    hh-rlhf failed: {e}")

    print(f"    Got {len(pairs)} safety pairs")
    return pairs[:n]


def load_direct_data(n: int) -> List[Tuple[str, str]]:
    """Short, direct responses — counter verbosity."""
    from datasets import load_dataset
    print(f"  Loading {n} direct response examples...")
    pairs = []

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    filler_starts = ["i'd be happy", "i would be happy", "sure!", "of course!",
                     "certainly!", "great question", "absolutely!"]
    for ex in ds:
        if len(pairs) >= n:
            break
        output = ex.get("output", "")
        instruction = ex.get("instruction", "")
        inp = ex.get("input", "")
        if inp:
            instruction = f"{instruction}\n\n{inp}"
        if (10 < len(output) < 200 and len(instruction) > 10
                and not any(output.lower().startswith(f) for f in filler_starts)):
            pairs.append((instruction, output))

    random.shuffle(pairs)
    print(f"    Got {len(pairs)} direct pairs")
    return pairs[:n]


def tokenize_pairs(pairs, tokenizer, max_seq_len, is_mc=False):
    """Tokenize instruction/response pairs into training examples.

    If is_mc=True, marks the answer token position and stores ABCD token IDs
    for contrastive MC loss (4-way softmax over options, not full vocab CE).
    """
    # Pre-compute ABCD token IDs (same for all MC examples)
    choice_ids = None
    if is_mc:
        choice_ids = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1]
                      for c in "ABCD"]

    examples = []
    skipped = 0
    for instruction, response in pairs:
        if not response or len(response.strip()) < 1:
            skipped += 1
            continue

        prompt = ALPACA_TEMPLATE.format(instruction=instruction)
        prompt_tokens = tokenizer.encode(prompt)
        response_tokens = tokenizer.encode(response, add_special_tokens=False)
        full_tokens = prompt_tokens + response_tokens

        if len(full_tokens) < 10:
            skipped += 1
            continue
        if len(full_tokens) > max_seq_len:
            full_tokens = full_tokens[:max_seq_len]

        actual_prompt_len = min(len(prompt_tokens), len(full_tokens))
        ex = {
            "tokens": full_tokens,
            "prompt_len": actual_prompt_len,
        }
        # For MC examples, record answer info for RBF and contrastive loss
        if is_mc and len(response_tokens) >= 1:
            ex["truth_token_id"] = response_tokens[0]
            ex["truth_position"] = actual_prompt_len  # position in targets array
            ex["choice_ids"] = choice_ids  # [A_id, B_id, C_id, D_id]
            # Which index (0-3) in choice_ids is the correct answer
            if response_tokens[0] in choice_ids:
                ex["correct_choice_idx"] = choice_ids.index(response_tokens[0])
            else:
                ex["correct_choice_idx"] = -1  # shouldn't happen
        examples.append(ex)

    return examples, skipped


def tokenize_letterblind(triples, tokenizer, max_seq_len):
    """Tokenize letter-blind MC examples for completion-contrastive training.

    Each triple is (prompt_text, option_texts, correct_idx).
    For each option, stores the FULL completion tokens. At training time,
    scores each option by sum logprob of its completion and uses N-way
    contrastive loss. No ABCD letter tokens in the loss at all.

    No collision issues — every option is scored by its full text, not
    just the first token. Handles any number of options (2 to 15+).
    """
    examples = []
    skipped = 0
    for prompt_text, option_texts, correct_idx in triples:
        prompt = ALPACA_TEMPLATE.format(instruction=prompt_text)
        prompt_tokens = tokenizer.encode(prompt)
        if len(prompt_tokens) > max_seq_len:
            skipped += 1
            continue

        # Get full completion tokens for each option
        option_completions = []
        valid = True
        max_allowed_comp = max_seq_len - len(prompt_tokens)
        for opt in option_texts:
            comp_tokens = tokenizer.encode(f" {opt}", add_special_tokens=False)
            if not comp_tokens:
                valid = False
                break
            # Truncate if needed (shouldn't happen often)
            option_completions.append(comp_tokens[:max_allowed_comp])

        if not valid:
            skipped += 1
            continue

        examples.append({
            "tokens": prompt_tokens,
            "prompt_len": len(prompt_tokens),
            "is_letterblind": True,
            "option_completions": option_completions,
            "correct_idx": correct_idx,
            "n_options": len(option_texts),
        })

    return examples, skipped


def load_mmlu_mc(n: int) -> List[Tuple[str, str]]:
    """MMLU auxiliary_train as MC training data — shuffled option order."""
    from datasets import load_dataset
    print(f"  Loading {n} MMLU MC examples...")
    ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    ds = ds.shuffle(seed=123)
    pairs = []
    choices = "ABCD"
    for ex in ds:
        if len(pairs) >= n:
            break
        q = ex["question"]
        options = ex.get("choices", [])
        answer_idx = ex.get("answer", -1)
        if not options or answer_idx < 0 or answer_idx >= len(options):
            continue
        n_opts = min(len(options), 4)
        order = list(range(n_opts))
        random.shuffle(order)
        prompt = q + "\n"
        mapped_correct = -1
        for j, oi in enumerate(order):
            prompt += f"{choices[j]}. {options[oi]}\n"
            if oi == answer_idx:
                mapped_correct = j
        prompt += "Answer:"
        if mapped_correct < 0:
            continue
        response = f" {choices[mapped_correct]}"
        pairs.append((prompt, response))
    print(f"    Got {len(pairs)} MMLU MC pairs")
    return pairs[:n]


def load_arc_mc(n: int) -> List[Tuple[str, str]]:
    """ARC-Challenge train as MC training data — shuffled option order."""
    from datasets import load_dataset
    print(f"  Loading {n} ARC MC examples...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    ds = ds.shuffle(seed=123)
    pairs = []
    choices = "ABCD"
    for ex in ds:
        if len(pairs) >= n:
            break
        q = ex["question"]
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        answer_key = ex["answerKey"]
        try:
            correct_idx = labels.index(answer_key)
        except ValueError:
            continue
        n_opts = min(len(labels), 4)
        order = list(range(n_opts))
        random.shuffle(order)
        prompt = q + "\n"
        mapped_correct = -1
        for j, oi in enumerate(order):
            prompt += f"{choices[j]}. {texts[oi]}\n"
            if oi == correct_idx:
                mapped_correct = j
        prompt += "Answer:"
        if mapped_correct < 0:
            continue
        response = f" {choices[mapped_correct]}"
        pairs.append((prompt, response))
    print(f"    Got {len(pairs)} ARC MC pairs")
    return pairs[:n]


def load_truthfulqa_mc(n: int) -> List[Tuple[str, str]]:
    """TruthfulQA as MC training data — 4 options, shuffled."""
    from datasets import load_dataset
    print(f"  Loading {n} TruthfulQA MC examples...")
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    ds = ds.shuffle(seed=42)
    pairs = []
    choices = "ABCD"
    for ex in ds:
        if len(pairs) >= n:
            break
        q = ex["question"]
        mc1 = ex.get("mc1_targets", {})
        choice_list = mc1.get("choices", [])
        labels = mc1.get("labels", [])
        if not choice_list or not labels:
            continue
        try:
            correct_idx = labels.index(1)
        except ValueError:
            continue
        correct_text = choice_list[correct_idx]
        wrong_texts = [choice_list[i] for i in range(len(choice_list)) if i != correct_idx]
        if len(wrong_texts) > 3:
            wrong_texts = wrong_texts[:3]
        if len(wrong_texts) < 3:
            continue  # Skip questions with < 4 options
        all_texts = [correct_text] + wrong_texts
        order = list(range(4))
        random.shuffle(order)
        prompt = q + "\n"
        mapped_correct = -1
        for j, oi in enumerate(order):
            prompt += f"{choices[j]}. {all_texts[oi]}\n"
            if oi == 0:
                mapped_correct = j
        prompt += "Answer:"
        if mapped_correct < 0:
            continue
        response = f" {choices[mapped_correct]}"
        pairs.append((prompt, response))
    print(f"    Got {len(pairs)} TruthfulQA MC pairs")
    return pairs[:n]


# ---------------------------------------------------------------------------
# Letter-blind MC loaders — return (prompt, option_texts, correct_idx) triples
# Loss operates on CONTENT tokens, never ABCD. Supports any # of options.
# ---------------------------------------------------------------------------

def load_mmlu_mc_lb(n: int):
    """MMLU letter-blind: keeps ABCD in prompt for readability, but loss
    will operate on content tokens only."""
    from datasets import load_dataset
    print(f"  Loading {n} MMLU letter-blind examples...")
    ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    ds = ds.shuffle(seed=123)
    triples = []
    choices = "ABCD"
    for ex in ds:
        if len(triples) >= n:
            break
        q = ex["question"]
        options = ex.get("choices", [])
        answer_idx = ex.get("answer", -1)
        if not options or answer_idx < 0 or answer_idx >= len(options):
            continue
        n_opts = min(len(options), 4)
        order = list(range(n_opts))
        random.shuffle(order)
        prompt = q + "\n"
        mapped_correct = -1
        option_texts = []
        for j, oi in enumerate(order):
            prompt += f"{choices[j]}. {options[oi]}\n"
            option_texts.append(options[oi])
            if oi == answer_idx:
                mapped_correct = j
        prompt += "Answer:"
        if mapped_correct < 0:
            continue
        triples.append((prompt, option_texts, mapped_correct))
    print(f"    Got {len(triples)} MMLU letter-blind triples")
    return triples[:n]


def load_arc_mc_lb(n: int):
    """ARC-Challenge letter-blind — supports 3-5 options."""
    from datasets import load_dataset
    print(f"  Loading {n} ARC letter-blind examples...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    ds = ds.shuffle(seed=123)
    triples = []
    choices = "ABCDE"
    for ex in ds:
        if len(triples) >= n:
            break
        q = ex["question"]
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        answer_key = ex["answerKey"]
        try:
            correct_idx = labels.index(answer_key)
        except ValueError:
            continue
        n_opts = len(labels)
        order = list(range(n_opts))
        random.shuffle(order)
        prompt = q + "\n"
        mapped_correct = -1
        option_texts = []
        for j, oi in enumerate(order):
            prompt += f"{choices[j]}. {texts[oi]}\n"
            option_texts.append(texts[oi])
            if oi == correct_idx:
                mapped_correct = j
        prompt += "Answer:"
        if mapped_correct < 0:
            continue
        triples.append((prompt, option_texts, mapped_correct))
    print(f"    Got {len(triples)} ARC letter-blind triples")
    return triples[:n]


def load_truthfulqa_mc_lb(n: int):
    """TruthfulQA letter-blind — ALL options (up to 15), not truncated to 4.

    This is the key fix: TruthfulQA has variable numbers of wrong answers.
    Previous code truncated to 4 total, potentially excluding the correct
    answer. Now includes ALL options with N-way contrastive loss.
    """
    from datasets import load_dataset
    print(f"  Loading {n} TruthfulQA letter-blind examples (ALL options)...")
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    ds = ds.shuffle(seed=42)
    triples = []
    choices = "ABCDEFGHIJKLMNO"  # Support up to 15 options
    for ex in ds:
        if len(triples) >= n:
            break
        q = ex["question"]
        mc1 = ex.get("mc1_targets", {})
        choice_list = mc1.get("choices", [])
        labels = mc1.get("labels", [])
        if not choice_list or not labels:
            continue
        try:
            correct_idx = labels.index(1)
        except ValueError:
            continue
        if len(choice_list) < 2:
            continue
        # Use ALL options — don't truncate
        n_opts = min(len(choice_list), len(choices))
        order = list(range(n_opts))
        random.shuffle(order)
        prompt = q + "\n"
        mapped_correct = -1
        option_texts = []
        for j, oi in enumerate(order):
            prompt += f"{choices[j]}. {choice_list[oi]}\n"
            option_texts.append(choice_list[oi])
            if oi == correct_idx:
                mapped_correct = j
        prompt += "Answer:"
        if mapped_correct < 0:
            continue
        triples.append((prompt, option_texts, mapped_correct))
    avg_opts = np.mean([len(t[1]) for t in triples]) if triples else 0
    print(f"    Got {len(triples)} TruthfulQA letter-blind triples "
          f"(avg {avg_opts:.1f} options)")
    return triples[:n]


# ---------------------------------------------------------------------------
# Gravitational truth — scan training data for known facts, add attractors
# ---------------------------------------------------------------------------

def load_truth_dict():
    """Load the truth dictionary from JSON."""
    dict_path = os.path.join(os.path.dirname(__file__), "truth_dict.json")
    with open(dict_path) as f:
        data = json.load(f)
    return data["truths"]


def scan_for_truth_anchors(tokens, tokenizer, truth_entries):
    """Scan tokenized text for truth dictionary matches.

    Returns list of (position, truth_token_id) tuples where the truth
    token appears in the response and should be gravitationally attracted.
    """
    # Decode full text for context matching
    text = tokenizer.decode(tokens).lower()
    anchors = []

    for entry in truth_entries:
        context = entry["context"].lower()
        if context not in text:
            continue
        # Find where the truth text appears in tokens
        truth_text = entry["truth"]
        truth_toks = tokenizer.encode(f" {truth_text}", add_special_tokens=False)
        if not truth_toks:
            continue
        first_truth_tok = truth_toks[0]

        # Search for the truth token AFTER the context appears
        context_pos = text.find(context)
        # Convert rough char position to token position (approximate)
        # by re-encoding up to context_pos
        prefix_tokens = tokenizer.encode(tokenizer.decode(tokens)[:context_pos + len(context)],
                                          add_special_tokens=False)
        search_start = max(len(prefix_tokens) - 5, 0)  # buffer for tokenization mismatch

        for pos in range(search_start, len(tokens)):
            if tokens[pos] == first_truth_tok:
                anchors.append((pos, first_truth_tok))
                break  # One anchor per truth entry

    return anchors


def tokenize_with_truth(pairs, tokenizer, max_seq_len, truth_entries):
    """Tokenize SFT pairs and scan for gravitational truth anchors."""
    examples = []
    skipped = 0
    n_anchored = 0
    total_anchors = 0

    for instruction, response in pairs:
        if not response or len(response.strip()) < 1:
            skipped += 1
            continue

        prompt = ALPACA_TEMPLATE.format(instruction=instruction)
        prompt_tokens = tokenizer.encode(prompt)
        response_tokens = tokenizer.encode(response, add_special_tokens=False)
        full_tokens = prompt_tokens + response_tokens

        if len(full_tokens) < 10:
            skipped += 1
            continue
        if len(full_tokens) > max_seq_len:
            full_tokens = full_tokens[:max_seq_len]

        actual_prompt_len = min(len(prompt_tokens), len(full_tokens))

        # Scan for truth anchors in the FULL text (context may be in prompt,
        # truth token in response)
        anchors = scan_for_truth_anchors(full_tokens, tokenizer, truth_entries)

        ex = {
            "tokens": full_tokens,
            "prompt_len": actual_prompt_len,
        }
        if anchors:
            ex["truth_anchors"] = anchors
            n_anchored += 1
            total_anchors += len(anchors)

        examples.append(ex)

    print(f"    Truth scanning: {n_anchored}/{len(examples)} examples have anchors "
          f"({total_anchors} total anchors)")
    return examples, skipped


def load_truth_probes(n: int) -> List[Tuple[str, str]]:
    """Generate training pairs directly from the truth dictionary.

    These are dedicated truth reinforcement examples — not from any dataset,
    just direct Q&A from the truth dictionary.
    """
    truth_entries = load_truth_dict()
    pairs = []

    # Generate question-answer pairs from truth entries
    templates = [
        ("What is the {context}?", "{truth}"),
        ("Tell me the {context}.", "The {context} is {truth}."),
        ("Name the {context}.", "{truth}"),
    ]

    for entry in truth_entries:
        ctx = entry["context"]
        truth = entry["truth"]
        for q_template, a_template in templates:
            q = q_template.format(context=ctx, truth=truth)
            a = a_template.format(context=ctx, truth=truth)
            pairs.append((q, a))

    random.shuffle(pairs)
    print(f"  Loading {min(n, len(pairs))} truth probe pairs from {len(truth_entries)} entries...")
    return pairs[:n]


# ---------------------------------------------------------------------------
# Data mix definitions
# ---------------------------------------------------------------------------
DATA_MIXES = {
    "sft_only": lambda smoke: {
        "sft": (load_sft_data, 150 if smoke else 3000),
        "safety": (load_safety_data, 30 if smoke else 500),
        "direct": (load_direct_data, 20 if smoke else 500),
    },
    "v13": lambda smoke: {
        "sft": (load_sft_data, 100 if smoke else 2000),
        "safety": (load_safety_data, 20 if smoke else 500),
        "direct": (load_direct_data, 20 if smoke else 300),
        "mmlu_mc": (load_mmlu_mc, 50 if smoke else 1000),
        "arc_mc": (load_arc_mc, 20 if smoke else 500),
        "truthfulqa_mc": (load_truthfulqa_mc, 20 if smoke else 400),
    },
    "letterblind": lambda smoke: {
        "sft": (load_sft_data, 100 if smoke else 2000),
        "safety": (load_safety_data, 20 if smoke else 500),
        "direct": (load_direct_data, 20 if smoke else 300),
        "mmlu_lb": (load_mmlu_mc_lb, 50 if smoke else 1000),
        "arc_lb": (load_arc_mc_lb, 20 if smoke else 500),
        "truthfulqa_lb": (load_truthfulqa_mc_lb, 20 if smoke else 400),
    },
    "gravitational": lambda smoke: {
        "sft": (load_sft_data, 100 if smoke else 2500),
        "safety": (load_safety_data, 20 if smoke else 500),
        "direct": (load_direct_data, 20 if smoke else 300),
        "truth_probes": (load_truth_probes, 50 if smoke else 500),
    },
}


def load_data(tokenizer, max_seq_len, smoke=False, data_mix="sft_only"):
    """Load and tokenize all data."""
    mix_fn = DATA_MIXES.get(data_mix, DATA_MIXES["sft_only"])
    mix = mix_fn(smoke)

    all_train, all_val = [], []
    stats = {}

    mc_sources = {"mmlu_mc", "arc_mc", "truthfulqa_mc"}
    lb_sources = {name for name in mix.keys() if name.endswith("_lb")}
    use_grav = data_mix == "gravitational" and GRAV_LAMBDA > 0
    truth_entries = load_truth_dict() if use_grav else []

    for name, (loader, n) in mix.items():
        t0 = time.time()

        if name in lb_sources:
            # Letter-blind MC: loader returns triples, dedicated tokenizer
            triples = loader(n)
            examples, skipped = tokenize_letterblind(triples, tokenizer, max_seq_len)
            n_loaded = len(triples)
        elif use_grav and name in ("sft", "direct", "truth_probes"):
            # Gravitational: tokenize with truth scanning
            pairs = loader(n)
            examples, skipped = tokenize_with_truth(
                pairs, tokenizer, max_seq_len, truth_entries)
            n_loaded = len(pairs)
        else:
            pairs = loader(n)
            is_mc = name in mc_sources
            examples, skipped = tokenize_pairs(pairs, tokenizer, max_seq_len, is_mc=is_mc)
            n_loaded = len(pairs)

        elapsed = time.time() - t0

        n_val = max(int(len(examples) * 0.02), 1)
        random.shuffle(examples)
        all_val.extend(examples[:n_val])
        all_train.extend(examples[n_val:])

        avg_len = sum(len(e["tokens"]) for e in examples) / max(len(examples), 1)
        stats[name] = {"loaded": n_loaded, "tokenized": len(examples),
                       "skipped": skipped, "avg_tokens": avg_len, "time": elapsed}
        lb_tag = " [letter-blind]" if name in lb_sources else ""
        print(f"    {name}: {len(examples)} examples, avg {avg_len:.0f} tok, "
              f"{elapsed:.1f}s{lb_tag}")

    random.seed(42)
    random.shuffle(all_train)
    random.shuffle(all_val)
    all_val = all_val[:200]  # Cap val

    print(f"\n  Total: {len(all_train)} train, {len(all_val)} val")
    return all_train, all_val, stats


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(model, tokenizer, adapter, train_examples, val_examples, *,
          epochs=3, lr=1e-4, save_dir=RESULTS_DIR, log_every=100, eval_every=500):
    """Train adapter. Returns best val loss."""
    lm_head = get_lm_head_fn(model)
    total_steps = len(train_examples) * epochs
    warmup_steps = max(int(total_steps * 0.05), 1)
    cos_steps = max(total_steps - warmup_steps, 1)

    warmup_sched = optim.linear_schedule(1e-7, lr, warmup_steps)
    cos_sched = optim.cosine_decay(lr, cos_steps)
    lr_schedule = optim.join_schedules([warmup_sched, cos_sched], [warmup_steps])
    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=0.01)

    # Compute base val loss (skip letterblind examples — no response tokens)
    base_losses = []
    for ex in val_examples[:50]:
        if ex.get("is_letterblind"):
            continue
        tokens = mx.array(ex["tokens"])[None, :]
        h = model.model(tokens[:, :-1])
        mx.eval(h)
        base_logits = lm_head(h)
        mx.eval(base_logits)
        targets = tokens[:, 1:]
        loss = nn.losses.cross_entropy(base_logits, targets, reduction="mean")
        mx.eval(loss)
        base_losses.append(float(loss))
    avg_base = np.mean(base_losses) if base_losses else 0.0
    print(f"\n  Base val CE: {avg_base:.4f} (ppl ~{np.exp(avg_base):.1f})")

    l2_penalty = L2_SHIFT_PENALTY
    rbf_lambda = RBF_LAMBDA
    rbf_sigma = RBF_SIGMA
    mc_contrastive = MC_CONTRASTIVE

    grav_lambda = GRAV_LAMBDA
    grav_sigma = GRAV_SIGMA

    def loss_fn(adapter, h, targets, mask, truth_token_id=-1, truth_pos=-1,
                choice_ids_arr=None, correct_choice_idx=-1,
                grav_positions=None, grav_token_ids=None):
        """Loss with optional contrastive MC, RBF, and gravitational truth terms.

        Gravitational truth: when training data contains known facts from the
        truth dictionary, adds a soft Gaussian attractor pulling the adapted
        logits toward the correct truth token at each anchor position.
        Unlike MC training (operates on ABCD letters), this operates on
        CONTENT tokens — the actual answers in natural text.
        """
        base_logits = lm_head(h)
        raw_shifts = adapter(base_logits)
        centered = raw_shifts - raw_shifts.mean(axis=-1, keepdims=True)
        combined = base_logits + centered
        adapted = LOGIT_SOFTCAP * mx.tanh(combined / LOGIT_SOFTCAP)
        ce = nn.losses.cross_entropy(adapted, targets, reduction="none")
        mask_sum = mx.maximum(mask.sum(), mx.array(1.0))
        ce_loss = (ce * mask).sum() / mask_sum

        total_loss = ce_loss

        # Contrastive MC loss: 4-way softmax at answer position
        if mc_contrastive and choice_ids_arr is not None and correct_choice_idx >= 0:
            pred_pos = truth_pos - 1
            if pred_pos >= 0 and pred_pos < adapted.shape[1]:
                logits_at_ans = adapted[0, pred_pos, :]
                option_logits = logits_at_ans[choice_ids_arr]  # [4]
                option_logits = option_logits[None, :]  # [1, 4]
                target_idx = mx.array([correct_choice_idx])
                mc_ce = nn.losses.cross_entropy(option_logits, target_idx,
                                                 reduction="mean")
                total_loss = total_loss + 0.5 * mc_ce

        # L2 penalty on shift magnitudes
        if l2_penalty > 0:
            shift_l2 = (centered * centered).mean()
            total_loss = total_loss + l2_penalty * shift_l2

        # Truth Sink RBF loss at answer position
        if rbf_lambda > 0 and truth_token_id >= 0 and truth_pos >= 0:
            pred_pos = truth_pos - 1
            if pred_pos >= 0 and pred_pos < adapted.shape[1]:
                logits_at_ans = adapted[0, pred_pos, :]
                correct_logit = logits_at_ans[truth_token_id]
                max_logit = mx.max(logits_at_ans)
                d = max_logit - correct_logit
                rbf_loss = rbf_lambda * (1.0 - mx.exp(-(d * d) / (rbf_sigma * rbf_sigma)))
                total_loss = total_loss + rbf_loss

        # Gravitational truth: attract adapted logits toward truth tokens
        # at positions where known facts appear in the training text
        if grav_lambda > 0 and grav_positions is not None:
            n_anchors = grav_positions.shape[0]
            for ai in range(n_anchors):
                pos = int(grav_positions[ai])
                tid = int(grav_token_ids[ai])
                # Position in adapted is shifted by 1 (predicting next token)
                pred_pos = pos - 1
                if pred_pos < 0 or pred_pos >= adapted.shape[1]:
                    continue
                logits_at_pos = adapted[0, pred_pos, :]
                truth_logit = logits_at_pos[tid]
                max_logit = mx.max(logits_at_pos)
                gap = max_logit - truth_logit
                # Gaussian well: 0 when gap=0, approaches grav_lambda as gap→∞
                grav_loss = grav_lambda * (1.0 - mx.exp(
                    -(gap * gap) / (2.0 * grav_sigma * grav_sigma)))
                total_loss = total_loss + grav_loss / max(n_anchors, 1)

        return total_loss

    def completion_loss_fn(adapter, h_batch, targets_batch, comp_mask,
                           correct_idx_arr):
        """Completion-based contrastive loss — fully letter-blind.

        Scores each option by the adapted sum logprob of its FULL completion
        text. N-way contrastive (softmax) over option scores.
        No ABCD letter tokens in the loss — position bias impossible.

        h_batch:      [N, T, D]  hidden states for all N options (padded)
        targets_batch: [N, T]    target tokens (padded, only comp positions matter)
        comp_mask:     [N, T]    1.0 for completion positions, 0.0 elsewhere
        correct_idx_arr: [1]     index of correct option
        """
        base_logits = lm_head(h_batch)  # [N, T, V]
        raw_shifts = adapter(base_logits)
        centered = raw_shifts - raw_shifts.mean(axis=-1, keepdims=True)
        combined = base_logits + centered
        adapted = LOGIT_SOFTCAP * mx.tanh(combined / LOGIT_SOFTCAP)

        # Per-token CE (negative log probs)
        ce = nn.losses.cross_entropy(
            adapted, targets_batch, reduction="none")  # [N, T]

        # Sum logprob per option (masked to completion tokens only)
        scores = -(ce * comp_mask).sum(axis=-1)  # [N] — higher = better
        scores = scores[None, :]  # [1, N]

        loss = nn.losses.cross_entropy(
            scores, correct_idx_arr, reduction="mean")

        if l2_penalty > 0:
            shift_l2 = (centered * centered).mean()
            loss = loss + l2_penalty * shift_l2

        return loss

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)
    comp_loss_and_grad = nn.value_and_grad(adapter, completion_loss_fn)

    best_val_loss = float("inf")
    global_step = 0
    log_file = os.path.join(save_dir, "training_log.jsonl")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        random.shuffle(train_examples)

        for i, ex in enumerate(train_examples):
            tokens = mx.array(ex["tokens"])[None, :]

            if ex.get("is_letterblind"):
                # Completion-contrastive: score full option text, not letters
                prompt_tokens = ex["tokens"]
                option_completions = ex["option_completions"]
                correct_idx = ex["correct_idx"]
                prompt_len = len(prompt_tokens)

                # Build padded batch: one sequence per option
                max_comp = max(len(c) for c in option_completions)
                max_input_len = prompt_len + max_comp - 1

                input_batch = []
                targets_batch = []
                mask_batch = []

                for comp in option_completions:
                    K = len(comp)
                    full_seq = prompt_tokens + comp
                    # Input: drop last token, pad to max_input_len
                    inp = full_seq[:-1] + [0] * (max_input_len - len(full_seq) + 1)
                    # Targets: shift right, pad
                    tgt = full_seq[1:] + [0] * (max_input_len - len(full_seq) + 1)
                    # Mask: 1 only for completion target positions
                    mask = ([0.0] * (prompt_len - 1)
                            + [1.0] * K
                            + [0.0] * (max_comp - K))
                    input_batch.append(inp)
                    targets_batch.append(tgt)
                    mask_batch.append(mask)

                input_arr = mx.array(input_batch)   # [N, T]
                h_batch = model.model(input_arr)     # [N, T, D]
                mx.eval(h_batch)

                targets_arr = mx.array(targets_batch)  # [N, T]
                mask_arr = mx.array(mask_batch)         # [N, T]
                cidx_arr = mx.array([correct_idx])

                loss, grads = comp_loss_and_grad(
                    adapter, h_batch, targets_arr, mask_arr, cidx_arr)
            else:
                # Standard SFT / old-style MC training
                prompt_len = ex["prompt_len"]
                h = model.model(tokens[:, :-1])
                mx.eval(h)

                targets = tokens[:, 1:]
                mask = mx.zeros(targets.shape, dtype=mx.float32)
                if prompt_len < targets.shape[1]:
                    mask[:, prompt_len:] = 1.0

                # Extract MC info for contrastive/RBF loss
                truth_token_id = ex.get("truth_token_id", -1)
                truth_pos = ex.get("truth_position", -1)
                correct_choice_idx = ex.get("correct_choice_idx", -1)
                choice_ids_list = ex.get("choice_ids", None)
                choice_ids_arr = mx.array(choice_ids_list) if choice_ids_list else None

                # Extract gravitational truth anchors
                anchors = ex.get("truth_anchors", None)
                grav_positions = None
                grav_token_ids = None
                if anchors and GRAV_LAMBDA > 0:
                    grav_positions = mx.array([a[0] for a in anchors])
                    grav_token_ids = mx.array([a[1] for a in anchors])

                loss, grads = loss_and_grad(adapter, h, targets, mask,
                                             truth_token_id=truth_token_id,
                                             truth_pos=truth_pos,
                                             choice_ids_arr=choice_ids_arr,
                                             correct_choice_idx=correct_choice_idx,
                                             grav_positions=grav_positions,
                                             grav_token_ids=grav_token_ids)

            # Gradient clipping — essential with short MC responses
            grads, grad_norm = optim.clip_grad_norm(grads, max_norm=1.0)

            # NaN guard — skip step if loss is NaN
            loss_val_check = float(loss)
            if np.isnan(loss_val_check) or np.isinf(loss_val_check):
                if global_step % log_every == 0:
                    print(f"  step {global_step:5d} | SKIPPED (NaN/Inf loss)")
                global_step += 1
                continue

            optimizer.update(adapter, grads)
            mx.eval(adapter.parameters(), optimizer.state)

            global_step += 1

            if global_step % log_every == 0:
                loss_val = float(loss)
                current_lr = float(lr_schedule(global_step))
                print(f"  step {global_step:5d} | epoch {epoch+1} [{i+1}/{len(train_examples)}] "
                      f"| loss {loss_val:.4f} | lr {current_lr:.2e}")
                with open(log_file, "a") as f:
                    f.write(json.dumps({"step": global_step, "loss": loss_val,
                                        "lr": current_lr, "epoch": epoch}) + "\n")

            if global_step % eval_every == 0:
                val_loss = _eval_val(model, adapter, lm_head, val_examples)
                print(f"  >>> val loss: {val_loss:.4f} (base: {avg_base:.4f}, "
                      f"delta={avg_base - val_loss:+.4f})")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    weights = dict(tree_flatten(adapter.parameters()))
                    mx.savez(os.path.join(save_dir, "best.npz"), **weights)
                    adapter.config.save(os.path.join(save_dir, "best_config.json"))
                    print(f"  >>> saved best model (val_loss={val_loss:.4f})")

    # Final save
    weights = dict(tree_flatten(adapter.parameters()))
    mx.savez(os.path.join(save_dir, "final.npz"), **weights)
    adapter.config.save(os.path.join(save_dir, "final_config.json"))

    return best_val_loss, avg_base


def _eval_val(model, adapter, lm_head, val_examples):
    """Quick validation loss (skips letterblind examples — no response tokens)."""
    losses = []
    for ex in val_examples[:100]:
        if ex.get("is_letterblind"):
            continue  # No response tokens to compute CE on
        tokens = mx.array(ex["tokens"])[None, :]
        prompt_len = ex["prompt_len"]
        h = model.model(tokens[:, :-1])
        mx.eval(h)
        base_logits = lm_head(h)
        adapted = apply_adapter(adapter, base_logits)
        targets = tokens[:, 1:]
        mask = mx.zeros(targets.shape, dtype=mx.float32)
        if prompt_len < targets.shape[1]:
            mask[:, prompt_len:] = 1.0
        ce = nn.losses.cross_entropy(adapted, targets, reduction="none")
        mask_sum_val = max(float(mask.sum()), 1.0)
        masked = (ce * mask).sum() / mask_sum_val
        mx.eval(masked)
        losses.append(float(masked))
    return np.mean(losses) if losses else 0.0


# ---------------------------------------------------------------------------
# Evaluation — MMLU (the one that matters)
# ---------------------------------------------------------------------------
def evaluate_mmlu(model, tokenizer, adapter, lm_head, n_questions=200,
                  shuffle_options=False):
    """Evaluate MMLU accuracy (logit-forced MC).

    If shuffle_options=True, randomizes option order at eval time.
    This separates content discrimination from position bias.
    """
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(min(n_questions, len(ds))))

    choices = "ABCD"
    choice_ids = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in choices]

    base_correct = 0
    adapter_correct = 0
    base_preds = [0, 0, 0, 0]
    adapter_preds = [0, 0, 0, 0]

    eval_rng = random.Random(999)  # Separate RNG for eval shuffling

    t0 = time.time()
    for idx, ex in enumerate(ds):
        original_answer = ex["answer"]
        option_texts = list(ex["choices"])

        if shuffle_options:
            # Shuffle option order and track where correct answer moved
            order = list(range(len(option_texts)))
            eval_rng.shuffle(order)
            shuffled_texts = [option_texts[o] for o in order]
            # Find where the original correct answer ended up
            mapped_answer = order.index(original_answer)
            option_texts = shuffled_texts
            correct_idx = mapped_answer
        else:
            correct_idx = original_answer

        prompt_text = f"{ex['question']}\n"
        for j, opt in enumerate(option_texts):
            prompt_text += f"{choices[j]}. {opt}\n"
        prompt_text += "Answer:"

        full_prompt = ALPACA_TEMPLATE.format(instruction=prompt_text)
        tokens = mx.array(tokenizer.encode(full_prompt))[None, :]

        h = model.model(tokens)
        mx.eval(h)
        base_logits = lm_head(h)
        mx.eval(base_logits)
        adapted = apply_adapter(adapter, base_logits)
        mx.eval(adapted)

        bl = base_logits[0, -1, :]
        al = adapted[0, -1, :]

        base_scores = [float(bl[cid]) for cid in choice_ids]
        adapted_scores = [float(al[cid]) for cid in choice_ids]

        bp = int(np.argmax(base_scores))
        ap = int(np.argmax(adapted_scores))

        if bp == correct_idx:
            base_correct += 1
        if ap == correct_idx:
            adapter_correct += 1
        base_preds[bp] += 1
        adapter_preds[ap] += 1

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{n_questions}] {elapsed:.0f}s  "
                  f"base={base_correct}/{idx+1}  adapter={adapter_correct}/{idx+1}")

    n = len(ds)
    elapsed = time.time() - t0
    base_acc = base_correct / n
    adapter_acc = adapter_correct / n
    delta = adapter_acc - base_acc

    mode = "SHUFFLED" if shuffle_options else "FIXED"
    print(f"\n{'=' * 60}")
    print(f"  MMLU RESULTS ({n} questions, {elapsed:.0f}s) [{mode} order]")
    print(f"{'=' * 60}")
    print(f"  Base:    {base_correct}/{n} = {base_acc:.1%}")
    print(f"  Adapter: {adapter_correct}/{n} = {adapter_acc:.1%}")
    print(f"  Delta:   {delta:+.1%}")
    print(f"  Base preds:    {base_preds}  ({[f'{p/n:.0%}' for p in base_preds]})")
    print(f"  Adapter preds: {adapter_preds}  ({[f'{p/n:.0%}' for p in adapter_preds]})")

    # Position bias check
    max_skew = max(adapter_preds) / (n / 4)
    if max_skew > 1.5:
        dominant = choices[adapter_preds.index(max(adapter_preds))]
        print(f"  *** WARNING: Position bias toward {dominant} (skew={max_skew:.1f}x) ***")

    return {"base": base_acc, "adapter": adapter_acc, "delta": delta,
            "base_preds": base_preds, "adapter_preds": adapter_preds,
            "n": n, "time": elapsed}


def evaluate_qualitative(model, tokenizer, adapter, n=5):
    """Generate a few responses to see if the adapter produces coherent text."""
    lm_head = get_lm_head_fn(model)

    prompts = [
        "What is the capital of France?",
        "Explain photosynthesis in one sentence.",
        "Write a haiku about the moon.",
        "What are three benefits of exercise?",
        "How do computers work?",
    ]

    print(f"\n{'=' * 60}")
    print(f"  QUALITATIVE SAMPLES")
    print(f"{'=' * 60}")

    results = []
    for prompt in prompts[:n]:
        full = ALPACA_TEMPLATE.format(instruction=prompt)
        tokens = tokenizer.encode(full)

        # Generate with adapter
        generated = list(tokens)
        for _ in range(100):
            inp = mx.array(generated)[None, :]
            h = model.model(inp)
            mx.eval(h)
            base_logits = lm_head(h)
            adapted = apply_adapter(adapter, base_logits)
            mx.eval(adapted)

            next_token = int(mx.argmax(adapted[0, -1, :]))
            if next_token == tokenizer.eos_token_id:
                break
            generated.append(next_token)

        response = tokenizer.decode(generated[len(tokens):])
        print(f"\n  Q: {prompt}")
        print(f"  A: {response[:200]}")
        results.append({"prompt": prompt, "response": response[:200]})

    return results


# ---------------------------------------------------------------------------
# Truth recall evaluation — test factual accuracy on truth dictionary
# ---------------------------------------------------------------------------
def evaluate_truth_recall(model, tokenizer, adapter, n_samples=60):
    """Test whether the adapter improves factual accuracy on truth dict items.

    Generates from truth dictionary questions and checks if the response
    contains the correct answer. Also compares logit preference (base vs adapted)
    for the truth token at the answer position.
    """
    lm_head = get_lm_head_fn(model)
    truth_entries = load_truth_dict()
    random.seed(777)
    random.shuffle(truth_entries)
    entries = truth_entries[:n_samples]

    base_correct = 0
    adapter_correct = 0
    base_logit_correct = 0
    adapter_logit_correct = 0
    results_detail = []

    print(f"\n{'=' * 60}")
    print(f"  TRUTH RECALL ({len(entries)} items)")
    print(f"{'=' * 60}")

    for entry in entries:
        ctx = entry["context"]
        truth = entry["truth"]
        question = f"What is the {ctx}?"
        full_prompt = ALPACA_TEMPLATE.format(instruction=question)
        tokens = tokenizer.encode(full_prompt)

        # Get truth token for logit comparison
        truth_toks = tokenizer.encode(f" {truth}", add_special_tokens=False)
        truth_first_tok = truth_toks[0] if truth_toks else -1

        # Logit check: does the model prefer the truth token at next position?
        inp = mx.array(tokens)[None, :]
        h = model.model(inp)
        mx.eval(h)
        base_logits = lm_head(h)
        mx.eval(base_logits)
        adapted = apply_adapter(adapter, base_logits)
        mx.eval(adapted)

        if truth_first_tok >= 0:
            bl = base_logits[0, -1, :]
            al = adapted[0, -1, :]
            base_top = int(mx.argmax(bl))
            adapter_top = int(mx.argmax(al))
            if base_top == truth_first_tok:
                base_logit_correct += 1
            if adapter_top == truth_first_tok:
                adapter_logit_correct += 1

        # Generation check: does the generated text contain the truth?
        generated = list(tokens)
        for _ in range(30):
            inp = mx.array(generated)[None, :]
            h = model.model(inp)
            mx.eval(h)
            bl = lm_head(h)
            al = apply_adapter(adapter, bl)
            mx.eval(al)
            next_token = int(mx.argmax(al[0, -1, :]))
            if next_token == tokenizer.eos_token_id:
                break
            generated.append(next_token)

        response = tokenizer.decode(generated[len(tokens):]).strip()
        if truth.lower() in response.lower():
            adapter_correct += 1

        # Base generation for comparison
        generated_base = list(tokens)
        for _ in range(30):
            inp = mx.array(generated_base)[None, :]
            h = model.model(inp)
            mx.eval(h)
            bl = lm_head(h)
            mx.eval(bl)
            next_token = int(mx.argmax(bl[0, -1, :]))
            if next_token == tokenizer.eos_token_id:
                break
            generated_base.append(next_token)

        response_base = tokenizer.decode(generated_base[len(tokens):]).strip()
        if truth.lower() in response_base.lower():
            base_correct += 1

        results_detail.append({
            "context": ctx, "truth": truth,
            "base_response": response_base[:100],
            "adapter_response": response[:100],
            "base_has_truth": truth.lower() in response_base.lower(),
            "adapter_has_truth": truth.lower() in response.lower(),
        })

    n = len(entries)
    print(f"  Generation — Base: {base_correct}/{n} ({base_correct/n:.1%})  "
          f"Adapter: {adapter_correct}/{n} ({adapter_correct/n:.1%})  "
          f"Delta: {(adapter_correct - base_correct)/n:+.1%}")
    print(f"  Logit pref — Base: {base_logit_correct}/{n} ({base_logit_correct/n:.1%})  "
          f"Adapter: {adapter_logit_correct}/{n} ({adapter_logit_correct/n:.1%})  "
          f"Delta: {(adapter_logit_correct - base_logit_correct)/n:+.1%}")

    # Show a few examples
    for d in results_detail[:5]:
        tag_b = "Y" if d["base_has_truth"] else "N"
        tag_a = "Y" if d["adapter_has_truth"] else "N"
        print(f"  [{tag_b}→{tag_a}] {d['context']}: truth='{d['truth']}' "
              f"base='{d['base_response'][:50]}' adapt='{d['adapter_response'][:50]}'")

    return {
        "base_gen_acc": base_correct / n,
        "adapter_gen_acc": adapter_correct / n,
        "gen_delta": (adapter_correct - base_correct) / n,
        "base_logit_acc": base_logit_correct / n,
        "adapter_logit_acc": adapter_logit_correct / n,
        "logit_delta": (adapter_logit_correct - base_logit_correct) / n,
        "n": n,
        "detail": results_detail[:10],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Toy-scale Operation Destroyer")
    parser.add_argument("--version", default="v1", help="Version tag")
    parser.add_argument("--smoke", action="store_true", help="Smoke test (200 examples, 1 epoch)")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--d_inner", type=int, default=D_INNER)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--mmlu_n", type=int, default=500)
    parser.add_argument("--softcap", type=float, default=LOGIT_SOFTCAP)
    parser.add_argument("--l2_penalty", type=float, default=L2_SHIFT_PENALTY,
                        help="L2 penalty on shift magnitudes (0=off, 0.01=moderate)")
    parser.add_argument("--rbf_lambda", type=float, default=RBF_LAMBDA,
                        help="Truth Sink RBF loss weight (0=off, 0.5=moderate)")
    parser.add_argument("--rbf_sigma", type=float, default=RBF_SIGMA,
                        help="RBF well width (smaller=narrower well, stronger pull)")
    parser.add_argument("--mc_contrastive", action="store_true",
                        help="Use 4-way contrastive loss for MC (prevents position bias)")
    parser.add_argument("--grav_lambda", type=float, default=GRAV_LAMBDA,
                        help="Gravitational truth loss weight (0=off, 0.1=moderate)")
    parser.add_argument("--grav_sigma", type=float, default=GRAV_SIGMA,
                        help="Gravitational well width in logit units (5=wide, 2=narrow)")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--data_mix", default="sft_only",
                        choices=list(DATA_MIXES.keys()),
                        help="Data mix to use")
    args = parser.parse_args()

    _update_softcap(args.softcap)
    _update_l2_penalty(args.l2_penalty)
    _update_rbf(args.rbf_lambda, args.rbf_sigma)
    _update_grav(args.grav_lambda, args.grav_sigma)
    global MC_CONTRASTIVE
    MC_CONTRASTIVE = args.mc_contrastive

    save_dir = os.path.join(RESULTS_DIR, args.version)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"  OPERATION DESTROYER — TOY SCALE ({args.model})")
    print(f"  Version: {args.version}{'  [SMOKE TEST]' if args.smoke else ''}")
    print("=" * 60)

    # Load model
    print(f"\nLoading {args.model}...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(args.model)
    model.freeze()
    lm_head = get_lm_head_fn(model)
    try:
        vocab_size = lm_head.weight.shape[0]
    except AttributeError:
        vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    print(f"  Loaded in {time.time()-t0:.1f}s: d_model={d_model}, vocab={vocab_size}")

    # Create adapter
    config = SnapOnConfig(
        d_model=d_model, d_inner=args.d_inner, n_layers=0,
        n_heads=8, mode="logit", vocab_size=vocab_size,
    )
    adapter = create_adapter(config)
    n_params = sum(p.size for _, p in tree_flatten(adapter.parameters()))
    print(f"  Adapter: {n_params:,} params ({n_params * 4 / 1e6:.1f} MB)")

    if args.eval_only:
        best_path = os.path.join(save_dir, "best.npz")
        if os.path.exists(best_path):
            weights = mx.load(best_path)
            adapter.load_weights(list(weights.items()))
            mx.eval(adapter.parameters())
            print("  Loaded best adapter weights")
        else:
            print(f"  WARNING: No checkpoint at {best_path}")
    else:
        # Load data
        print(f"\n{'=' * 60}")
        print(f"  LOADING DATA")
        print(f"{'=' * 60}")
        train_examples, val_examples, data_stats = load_data(
            tokenizer, MAX_SEQ_LEN, smoke=args.smoke, data_mix=args.data_mix)

        with open(os.path.join(save_dir, "data_stats.json"), "w") as f:
            json.dump(data_stats, f, indent=2, default=str)

        # Train
        epochs = 1 if args.smoke else args.epochs
        log_every = 50 if args.smoke else 100
        eval_every = 100 if args.smoke else 500

        print(f"\n{'=' * 60}")
        print(f"  TRAINING ({epochs} epoch{'s' if epochs > 1 else ''}, "
              f"{len(train_examples)} examples = {len(train_examples) * epochs} steps)")
        print(f"{'=' * 60}")

        t0 = time.time()
        best_val_loss, avg_base = train(
            model, tokenizer, adapter, train_examples, val_examples,
            epochs=epochs, lr=args.lr, save_dir=save_dir,
            log_every=log_every, eval_every=eval_every,
        )
        train_time = time.time() - t0
        print(f"\n  Training done in {train_time/60:.1f} min")

        # Load best
        best_path = os.path.join(save_dir, "best.npz")
        if os.path.exists(best_path):
            weights = mx.load(best_path)
            adapter.load_weights(list(weights.items()))
            mx.eval(adapter.parameters())

    # Evaluate
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION (fixed order)")
    print(f"{'=' * 60}")

    mmlu_n = 50 if args.smoke else args.mmlu_n
    mmlu_result = evaluate_mmlu(model, tokenizer, adapter, lm_head,
                                n_questions=mmlu_n, shuffle_options=False)

    # Also run shuffled eval to separate content from position
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION (shuffled order — position-bias free)")
    print(f"{'=' * 60}")
    mmlu_shuffled = evaluate_mmlu(model, tokenizer, adapter, lm_head,
                                  n_questions=mmlu_n, shuffle_options=True)

    qual_result = evaluate_qualitative(model, tokenizer, adapter)

    # Truth recall (only if gravitational mode was used)
    truth_recall = None
    if args.data_mix == "gravitational" or args.grav_lambda > 0:
        truth_recall = evaluate_truth_recall(model, tokenizer, adapter)

    # Save results
    results = {
        "version": args.version,
        "model": args.model,
        "d_inner": args.d_inner,
        "n_params": n_params,
        "softcap": args.softcap,
        "l2_penalty": args.l2_penalty,
        "rbf_lambda": args.rbf_lambda,
        "rbf_sigma": args.rbf_sigma,
        "grav_lambda": args.grav_lambda,
        "grav_sigma": args.grav_sigma,
        "data_mix": args.data_mix,
        "lr": args.lr,
        "smoke": args.smoke,
        "mmlu_fixed": mmlu_result,
        "mmlu_shuffled": mmlu_shuffled,
        "qualitative": qual_result,
        "truth_recall": truth_recall,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Verdict — use SHUFFLED results as the ground truth
    print(f"\n{'=' * 60}")
    print(f"  VERDICT")
    print(f"{'=' * 60}")
    delta_fixed = mmlu_result["delta"]
    delta_shuffled = mmlu_shuffled["delta"]
    max_pred = max(mmlu_result["adapter_preds"])
    n = mmlu_result["n"]

    print(f"  Fixed order:    delta={delta_fixed:+.1%}")
    print(f"  Shuffled order: delta={delta_shuffled:+.1%}  ← TRUE measure")

    issues = []
    if delta_shuffled < -0.05:
        issues.append(f"MMLU regression (shuffled): {delta_shuffled:+.1%}")
    if max_pred > n * 0.4:
        issues.append(f"Position bias (fixed): {max_pred}/{n} on one letter")

    if issues:
        print(f"  FAIL: {'; '.join(issues)}")
        print(f"  Do NOT proceed to full run.")
    else:
        print(f"  PASS: MMLU delta(shuffled)={delta_shuffled:+.1%}, no position bias")
        if args.smoke:
            print(f"  Safe to proceed to full run.")
        else:
            print(f"  Ready to test transfer to Qwen3-4B.")

    print(f"\n  Results saved to {save_dir}/")


if __name__ == "__main__":
    main()
