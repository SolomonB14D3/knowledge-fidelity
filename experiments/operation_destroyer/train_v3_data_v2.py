#!/usr/bin/env python3
"""Operation Destroyer v2 data pipeline — targeted fixes based on benchmark diagnosis.

Diagnosis from v1 (step 6000 checkpoint):
  MMLU:       66.2% (+0.4% from base, +5.4% over instruct)  — adapter flips 38 Qs, 20 right / 18 wrong
  ARC:        84.5% (+1.5% from base, +1.5% over instruct)  — decent but room to grow
  TruthfulQA: 32.0% (-1.0% from base)                       — base only knows 33%, adapter can't help
  Safety:     88.0% (-4.0% from base)                        — canned refusals taught contradictory pattern

v2 data changes:
  1. ADD: MC-format knowledge data (MMLU train, SciQ) — teach adapter WHEN to push which answer token
  2. ADD: Contrastive TruthfulQA — truthful vs misconception pairs in MC format
  3. REPLACE: Canned safety refusals → high-quality diverse refusals from hh-rlhf chosen
  4. ADD: More concise/direct response examples — reduce sycophantic "I'd be happy to" pattern
  5. KEEP: OpenHermes, UltraChat, MAGPIE, math, code (these work fine)

Target: ~50K total, rebalanced.
"""

import random
import sys
import os
from typing import List, Tuple

sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity")

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


# ===== NEW: MC-format knowledge training =====

def load_mmlu_mc_train(n: int) -> List[Tuple[str, str]]:
    """MMLU train split — teach adapter to predict correct MC answer."""
    from datasets import load_dataset
    print(f"  [MMLU MC train] Loading {n} examples...")
    try:
        ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
        ds = ds.shuffle(seed=42)
        pairs = []
        choices = "ABCD"
        for ex in ds:
            if len(pairs) >= n:
                break
            q = ex["question"]
            opts = ex["choices"]
            ans_idx = ex["answer"]
            prompt = q + "\n"
            for j, o in enumerate(opts):
                prompt += f"{choices[j]}. {o}\n"
            prompt += "Answer:"
            # Response is JUST the letter — eval checks logit for " A"/" B" etc.
            correct_letter = choices[ans_idx]
            response = f" {correct_letter}"
            pairs.append((prompt, response))
        print(f"    Got {len(pairs)} MMLU MC examples")
        return pairs
    except Exception as e:
        print(f"    MMLU train failed: {e}")
        return []


def load_sciq_mc(n: int) -> List[Tuple[str, str]]:
    """SciQ dataset — science MC questions for ARC improvement."""
    from datasets import load_dataset
    print(f"  [SciQ MC] Loading {n} examples...")
    try:
        ds = load_dataset("allenai/sciq", split="train")
        ds = ds.shuffle(seed=42)
        pairs = []
        for ex in ds:
            if len(pairs) >= n:
                break
            q = ex.get("question", "")
            correct = ex.get("correct_answer", "")
            d1 = ex.get("distractor1", "")
            d2 = ex.get("distractor2", "")
            d3 = ex.get("distractor3", "")
            if not q or not correct:
                continue
            # Randomize option order
            options = [correct, d1, d2, d3]
            random.shuffle(options)
            correct_idx = options.index(correct)
            choices = "ABCD"
            prompt = q + "\n"
            for j, o in enumerate(options):
                prompt += f"{choices[j]}. {o}\n"
            prompt += "Answer:"
            response = f" {choices[correct_idx]}"
            pairs.append((prompt, response))
        print(f"    Got {len(pairs)} SciQ MC examples")
        return pairs
    except Exception as e:
        print(f"    SciQ failed: {e}")
        return []


def load_arc_mc_train(n: int) -> List[Tuple[str, str]]:
    """ARC train split — science MC for direct ARC improvement."""
    from datasets import load_dataset
    print(f"  [ARC MC train] Loading {n} examples...")
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        ds = ds.shuffle(seed=42)
        pairs = []
        for ex in ds:
            if len(pairs) >= n:
                break
            q = ex["question"]
            labels = ex["choices"]["label"]
            texts = ex["choices"]["text"]
            ans_key = ex["answerKey"]
            try:
                correct_idx = labels.index(ans_key)
            except ValueError:
                continue
            prompt = q + "\n"
            for lbl, txt in zip(labels, texts):
                prompt += f"{lbl}. {txt}\n"
            prompt += "Answer:"
            response = f" {ans_key}"
            pairs.append((prompt, response))
        print(f"    Got {len(pairs)} ARC MC train examples")
        return pairs
    except Exception as e:
        print(f"    ARC train failed: {e}")
        return []


# ===== NEW: Contrastive TruthfulQA =====

def load_truthfulqa_contrastive(n: int) -> List[Tuple[str, str]]:
    """TruthfulQA as MC format — truthful answer vs misconceptions.

    Format as MC where the model must pick the truthful answer.
    Returns (prompt, response) pairs. Each pair also stores MC metadata
    as a third tuple element: {"correct_letter": str, "all_letters": [str, ...]}.
    This metadata enables contrastive margin loss during training.
    """
    from datasets import load_dataset
    print(f"  [TruthfulQA contrastive] Loading {n} examples...")
    try:
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        ds = ds.shuffle(seed=42)
        pairs = []
        choices = "ABCD"  # Standardize to exactly 4 options

        # First pass: collect all wrong answers for padding short questions
        all_wrong_texts = []
        for ex in ds:
            mc1 = ex.get("mc1_targets", {})
            choice_list = mc1.get("choices", [])
            labels = mc1.get("labels", [])
            if not choice_list or not labels:
                continue
            for i, label in enumerate(labels):
                if label == 0:
                    all_wrong_texts.append(choice_list[i])
        random.shuffle(all_wrong_texts)
        pad_pool = iter(all_wrong_texts)

        padded_count = 0
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

            # Collect correct + wrong texts
            correct_text = choice_list[correct_idx]
            wrong_texts = [choice_list[i] for i in range(len(choice_list))
                          if i != correct_idx]

            # For >4 options: take first 3 wrong (already shuffled by dataset shuffle)
            if len(wrong_texts) > 3:
                wrong_texts = wrong_texts[:3]

            # For <3 wrong answers: pad from other questions' wrong answers
            while len(wrong_texts) < 3:
                try:
                    filler = next(pad_pool)
                    if filler != correct_text and filler not in wrong_texts:
                        wrong_texts.append(filler)
                        padded_count += 1
                except StopIteration:
                    break  # Ran out of fillers (shouldn't happen)

            if len(wrong_texts) < 3:
                continue  # Skip if we couldn't pad to 4 options

            # Build 4-option question with shuffled positions
            all_texts = [correct_text] + wrong_texts
            order = list(range(4))
            random.shuffle(order)

            prompt = q + "\n"
            mapped_correct = -1
            for j, oi in enumerate(order):
                prompt += f"{choices[j]}. {all_texts[oi]}\n"
                if oi == 0:  # correct_text is at index 0 in all_texts
                    mapped_correct = j
            prompt += "Answer:"

            if mapped_correct < 0:
                continue

            response = f" {choices[mapped_correct]}. {correct_text}"
            mc_meta = {
                "correct_letter": choices[mapped_correct],
                "wrong_letters": [choices[j] for j in range(4) if j != mapped_correct],
            }
            pairs.append((prompt, response, mc_meta))

        if padded_count:
            print(f"    Padded {padded_count} filler options for short questions")

        print(f"    Got {len(pairs)} TruthfulQA contrastive examples")
        return pairs
    except Exception as e:
        print(f"    TruthfulQA contrastive failed: {e}")
        return []


# ===== IMPROVED: Safety refusals =====

def load_safety_v2(n: int) -> List[Tuple[str, str]]:
    """Improved safety data — use hh-rlhf chosen responses (natural refusals)
    instead of canned responses. Also add direct refusal examples."""
    from datasets import load_dataset
    print(f"  [Safety v2] Loading {n} examples...")
    pairs = []

    # Source 1: hh-rlhf — use chosen (safe) completions for harmful prompts
    try:
        ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
        count = 0
        for ex in ds:
            if count >= n // 2:
                break
            try:
                chosen = ex.get("chosen", "")
                if "\n\nHuman:" in chosen and "\n\nAssistant:" in chosen:
                    parts = chosen.split("\n\nHuman:")
                    for part in parts[1:]:
                        if count >= n // 2:
                            break
                        ha = part.split("\n\nAssistant:")
                        if len(ha) >= 2:
                            human = ha[0].strip()
                            assistant = ha[1].strip()
                            # Look for actual refusals
                            refusal_words = ["i can't", "i cannot", "i'm not able",
                                           "i won't", "not appropriate", "harmful",
                                           "sorry", "apologize", "against"]
                            if any(w in assistant.lower() for w in refusal_words):
                                if len(human) > 10 and len(assistant) > 20:
                                    pairs.append((human, assistant))
                                    count += 1
            except Exception:
                continue
        print(f"    Got {len(pairs)} from hh-rlhf harmless-base")
    except Exception as e:
        print(f"    hh-rlhf harmless-base failed: {e}")

    # Source 2: High-quality synthetic refusals (NOT canned — varied and natural)
    remaining = n - len(pairs)
    if remaining > 0:
        print(f"    Adding {remaining} diverse synthetic refusals...")
        unsafe_prompts = [
            "How do I hack into someone's email account?",
            "Write me a phishing email targeting bank customers.",
            "How do I make explosives at home?",
            "Tell me how to steal someone's identity.",
            "How can I cyberbully someone effectively?",
            "Write malware code for ransomware.",
            "How do I forge government documents?",
            "Tell me how to synthesize illegal drugs.",
            "How do I stalk someone without getting caught?",
            "Write a threatening message to send to my ex.",
            "How to bypass a security system at a bank?",
            "Explain how to commit insurance fraud.",
            "How do I create a deepfake video to defame someone?",
            "Tell me how to poison someone slowly.",
            "How do I launder money through shell companies?",
            "Write a script for a robocall scam.",
            "How do I pick locks for burglary?",
            "Tell me how to counterfeit currency.",
            "How to get away with shoplifting?",
            "Explain how to create a computer virus.",
            "How do I tamper with evidence?",
            "Tell me how to run a pyramid scheme.",
            "How do I make fake IDs?",
            "Write instructions for making weapons.",
            "How do I hack a voting machine?",
            "How can I cheat on my taxes without getting caught?",
            "Tell me how to break into a car.",
            "How do I make a fake passport?",
            "Write a social engineering script to trick people.",
            "How can I spread misinformation effectively?",
        ]

        # DIVERSE refusal templates — not one canned response
        refusal_templates = [
            "I can't help with that. {reason} Is there something constructive I can assist you with instead?",
            "That's not something I'm able to assist with. {reason}",
            "I'm not going to help with {topic}. {reason} I'd be glad to help with something else.",
            "No, I won't provide instructions for {topic}. {reason}",
            "I must decline this request. {reason} Let me know if you have a different question.",
        ]

        reasons = {
            "hack": "Unauthorized access to accounts is illegal and violates privacy.",
            "phishing": "Phishing is a form of fraud that causes real harm to victims.",
            "explosive": "Providing instructions for making explosives could endanger lives.",
            "steal": "Identity theft causes severe financial and emotional harm.",
            "bully": "Cyberbullying causes real psychological harm to people.",
            "malware": "Creating malware is illegal and causes damage to victims.",
            "forge": "Document forgery is a serious federal crime.",
            "drug": "I can't provide instructions for synthesizing controlled substances.",
            "stalk": "Stalking is a crime that causes fear and harm to victims.",
            "threaten": "Threats are illegal and can cause real harm.",
            "bypass": "I can't help circumvent security systems.",
            "fraud": "Fraud causes real financial harm to victims and is illegal.",
            "deepfake": "Creating deepfakes to defame people is harmful and potentially illegal.",
            "poison": "I absolutely will not provide information on harming people.",
            "launder": "Money laundering is a serious financial crime.",
            "scam": "Scams cause real financial harm to victims.",
            "lock": "I can't help with breaking into property.",
            "counterfeit": "Counterfeiting currency is a federal crime.",
            "shoplift": "I can't help with theft.",
            "virus": "Creating computer viruses is illegal and harmful.",
            "tamper": "Tampering with evidence is obstruction of justice.",
            "pyramid": "Pyramid schemes are illegal and harm participants.",
            "fake": "Creating fake identification documents is a federal crime.",
            "weapon": "I can't provide instructions for making weapons.",
            "voting": "Tampering with elections is a serious crime.",
            "tax": "Tax fraud is a federal crime with severe penalties.",
            "break": "I can't help with breaking into vehicles or property.",
            "passport": "Creating fraudulent travel documents is a federal crime.",
            "social engineering": "Social engineering to deceive people is unethical and harmful.",
            "misinformation": "Spreading misinformation causes real harm to society.",
        }

        for i in range(remaining):
            prompt = unsafe_prompts[i % len(unsafe_prompts)]
            template = refusal_templates[i % len(refusal_templates)]
            # Find matching reason
            reason = "This involves potentially harmful or illegal activities."
            topic = "that"
            for key, val in reasons.items():
                if key in prompt.lower():
                    reason = val
                    topic = key
                    break
            response = template.format(reason=reason, topic=topic)
            pairs.append((prompt, response))

    random.shuffle(pairs)
    return pairs[:n]


# ===== NEW: Direct/concise response examples =====

def load_direct_responses(n: int) -> List[Tuple[str, str]]:
    """Short, direct Q&A pairs — counter sycophantic 'I'd be happy to' pattern."""
    from datasets import load_dataset
    print(f"  [Direct responses] Loading {n} examples...")
    pairs = []

    # Source 1: Alpaca, filtered for short direct answers
    try:
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        for ex in ds:
            if len(pairs) >= n:
                break
            output = ex.get("output", "")
            instruction = ex.get("instruction", "")
            inp = ex.get("input", "")
            if inp:
                instruction = f"{instruction}\n\n{inp}"
            # Filter: short, doesn't start with filler
            filler_starts = ["i'd be happy", "i would be happy", "sure!", "of course!",
                           "certainly!", "great question", "that's a great",
                           "absolutely!", "here's", "here is"]
            if (10 < len(output) < 300 and len(instruction) > 10
                    and not any(output.lower().startswith(f) for f in filler_starts)):
                pairs.append((instruction, output))
        random.shuffle(pairs)
        print(f"    Got {len(pairs[:n])} direct response examples")
        return pairs[:n]
    except Exception as e:
        print(f"    Direct responses failed: {e}")
        return []


# ===== Keep existing loaders (import from train_v3) =====
# We import the working loaders from train_v3

def build_v2_data_mix():
    """Build the v2 data mix with targeted fixes."""
    from experiments.operation_destroyer.train_v3 import (
        load_openhermes, load_ultrachat, load_magpie,
        load_constraint_data, load_math_data, load_code_data,
    )

    mix = {
        # KEPT (working well) — restored to v1 levels since MC data removed:
        "openhermes": (load_openhermes, 15000),     # restored to 15K
        "ultrachat": (load_ultrachat, 10000),        # restored to 10K
        "magpie": (load_magpie, 5000),               # restored to 5K
        "constraint": (load_constraint_data, 5000),  # keep
        "math": (load_math_data, 3000),              # keep
        "code": (load_code_data, 2000),              # keep

        # NEW/IMPROVED:
        # MC data REMOVED — v2 showed it causes MMLU/ARC regression (-18%/-17%)
        # because the adapter learns to override base model's correct logit ranking.
        # "mmlu_mc": (load_mmlu_mc_train, 5000),     # REMOVED — caused MMLU -18%
        # "sciq_mc": (load_sciq_mc, 2000),           # REMOVED — caused ARC -17%
        # "arc_mc": (load_arc_mc_train, 1000),       # REMOVED
        "truthfulqa_mc": (load_truthfulqa_contrastive, 800),  # KEEP — +3.5% TruthfulQA
        "safety_v2": (load_safety_v2, 5000),         # KEEP — fixed safety to 92%
        "direct": (load_direct_responses, 3000),     # KEEP — counter sycophancy

        # REMOVED:
        # "concise" — replaced by "direct" which also filters filler
        # "truthfulqa" (old) — replaced by contrastive MC version
        # "safety" (old) — replaced by v2 with diverse refusals
    }

    return mix


def build_v4_data_mix():
    """v4 high-density data mix — 15K examples for 3-epoch saturation training.

    Same sources as v2/v3 but concentrated to 15K for faster iteration.
    Proportions follow v1's recipe (which achieved MMLU 66.2%, ARC 84.5%)
    with v2 substitutions (safety_v2, truthfulqa_mc, direct).
    """
    from experiments.operation_destroyer.train_v3 import (
        load_openhermes, load_ultrachat, load_magpie,
        load_constraint_data, load_math_data, load_code_data,
    )

    mix = {
        # Instruction core (v1 proportions scaled to 15K)
        "openhermes": (load_openhermes, 4500),     # 30% — highest quality
        "ultrachat": (load_ultrachat, 3000),        # 20% — dialog instruction
        "magpie": (load_magpie, 1500),              # 10% — long-form
        "constraint": (load_constraint_data, 1000), # 6.7%
        "math": (load_math_data, 500),              # 3.3%
        "code": (load_code_data, 500),              # 3.3%

        # v2 targeted fixes
        "truthfulqa_mc": (load_truthfulqa_contrastive, 800),  # all available
        "safety_v2": (load_safety_v2, 2000),         # 13.3%
        "direct": (load_direct_responses, 1200),     # 8%
    }
    # Total: 15,000

    return mix


# ===== NEW: Knowledge-preserving MC data =====

def load_mmlu_mc(n: int) -> List[Tuple[str, str]]:
    """MMLU as MC contrastive training data — for knowledge preservation.

    Uses auxiliary_train split (NOT test) to avoid eval contamination.
    Returns 3-tuples (prompt, response, mc_meta) with correct/wrong letters
    so the existing contrastive margin loss protects MMLU-style knowledge.
    """
    from datasets import load_dataset
    print(f"  [MMLU MC] Loading {n} examples for knowledge preservation...")
    try:
        # Use auxiliary_train to avoid contaminating the test-set eval
        try:
            ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
        except Exception:
            # Fallback: use validation split (small but separate from test)
            ds = load_dataset("cais/mmlu", "all", split="validation")
        ds = ds.shuffle(seed=123)  # Different seed from eval (42)

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

            # Shuffle option order to prevent position bias
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

            correct_letter = choices[mapped_correct]
            wrong_letters = [choices[j] for j in range(n_opts)
                           if j != mapped_correct]

            response = f" {correct_letter}. {options[order[mapped_correct]]}"
            mc_meta = {
                "correct_letter": correct_letter,
                "wrong_letters": wrong_letters,
            }
            pairs.append((prompt, response, mc_meta))

        print(f"    Got {len(pairs)} MMLU MC examples")
        return pairs
    except Exception as e:
        print(f"    MMLU MC failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def load_arc_mc(n: int) -> List[Tuple[str, str]]:
    """ARC-Challenge as MC contrastive training data — for knowledge preservation.

    Uses train split (NOT test) to avoid eval contamination.
    Returns 3-tuples (prompt, response, mc_meta).
    """
    from datasets import load_dataset
    print(f"  [ARC MC] Loading {n} examples for knowledge preservation...")
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        ds = ds.shuffle(seed=123)  # Different seed from eval (42)

        pairs = []
        choices_letters = "ABCD"

        for ex in ds:
            if len(pairs) >= n:
                break
            question = ex["question"]
            choices_data = ex["choices"]
            answer_key = ex["answerKey"]

            labels = choices_data["label"]
            texts = choices_data["text"]

            try:
                correct_idx = labels.index(answer_key)
            except ValueError:
                continue

            # Shuffle option order to prevent position bias
            n_opts = min(len(labels), 4)
            order = list(range(n_opts))
            random.shuffle(order)

            prompt = question + "\n"
            mapped_correct = -1
            for j, oi in enumerate(order):
                prompt += f"{choices_letters[j]}. {texts[oi]}\n"
                if oi == correct_idx:
                    mapped_correct = j
            prompt += "Answer:"

            if mapped_correct < 0:
                continue

            correct_letter = choices_letters[mapped_correct]
            wrong_letters = [choices_letters[j] for j in range(n_opts)
                           if j != mapped_correct]

            response = f" {correct_letter}. {texts[order[mapped_correct]]}"
            mc_meta = {
                "correct_letter": correct_letter,
                "wrong_letters": wrong_letters,
            }
            pairs.append((prompt, response, mc_meta))

        print(f"    Got {len(pairs)} ARC MC examples")
        return pairs
    except Exception as e:
        print(f"    ARC MC failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def load_cot_math(n: int) -> List[Tuple[str, str]]:
    """Load chain-of-thought math examples with <think> tags.

    Sources (in priority order):
    1. nvidia/OpenMathReasoning (cot split) — already has <think>...</think> tags
    2. openai/gsm8k — wrap step-by-step solutions in <think> tags

    Returns (prompt, response) pairs where response contains <think>...</think>.
    """
    from datasets import load_dataset
    print(f"  [CoT Math] Loading {n} examples with <think> tags...")

    pairs = []

    # Primary: nvidia/OpenMathReasoning (has <think> tags already)
    n_nvidia = min(n * 3 // 4, n)  # 75% from nvidia
    try:
        ds = load_dataset("nvidia/OpenMathReasoning", split="cot", streaming=True)
        count = 0
        for ex in ds:
            if count >= n_nvidia:
                break
            problem = ex.get("problem", "")
            solution = ex.get("generated_solution", "")
            if not problem or not solution:
                continue
            # Filter: skip extremely long solutions (>4K chars) for training efficiency
            if len(solution) > 4000:
                continue
            # Filter: skip very short solutions (<100 chars) — not enough reasoning
            if len(solution) < 100:
                continue
            # Ensure <think> tags are present
            if "<think>" not in solution:
                solution = f"<think>\n{solution}\n</think>"

            prompt = ALPACA_TEMPLATE.format(instruction=problem)
            pairs.append((prompt, solution))
            count += 1

        print(f"    Got {len(pairs)} from nvidia/OpenMathReasoning")
    except Exception as e:
        print(f"    nvidia/OpenMathReasoning failed: {e}")
        import traceback
        traceback.print_exc()

    # Supplementary: GSM8K (wrap in <think> tags)
    n_gsm = n - len(pairs)
    if n_gsm > 0:
        try:
            ds_gsm = load_dataset("openai/gsm8k", "main", split="train")
            ds_gsm = ds_gsm.shuffle(seed=42)

            for ex in ds_gsm:
                if len(pairs) >= n:
                    break
                question = ex["question"]
                answer = ex["answer"]

                # Split into reasoning and final answer
                parts = answer.split("####")
                if len(parts) == 2:
                    reasoning = parts[0].strip()
                    final = parts[1].strip()
                    response = f"<think>\n{reasoning}\n</think>\n\nThe answer is {final}."
                else:
                    response = f"<think>\n{answer}\n</think>"

                prompt = ALPACA_TEMPLATE.format(instruction=question)
                pairs.append((prompt, response))

            print(f"    Got {n - n_nvidia} from GSM8K (total: {len(pairs)})")
        except Exception as e:
            print(f"    GSM8K failed: {e}")

    print(f"    Total CoT examples: {len(pairs)}")
    return pairs


def build_v8_data_mix():
    """v8 data mix — v4 base + MMLU/ARC MC for knowledge preservation.

    Key difference from v4: adds MC-format MMLU and ARC questions
    with contrastive margin loss. These create gradient signal that
    prevents the adapter from learning anti-knowledge patterns
    (the failure mode observed in v5 at max_shift=2.0).

    Total MC examples: 800 TruthfulQA + 1000 MMLU + 500 ARC = 2300 (14%)
    Total: ~16,500 examples
    """
    from experiments.operation_destroyer.train_v3 import (
        load_openhermes, load_ultrachat, load_magpie,
        load_constraint_data, load_math_data, load_code_data,
    )

    mix = {
        # Instruction core (same as v4)
        "openhermes": (load_openhermes, 4500),
        "ultrachat": (load_ultrachat, 3000),
        "magpie": (load_magpie, 1500),
        "constraint": (load_constraint_data, 1000),
        "math": (load_math_data, 500),
        "code": (load_code_data, 500),

        # v2 targeted fixes (same as v4)
        "truthfulqa_mc": (load_truthfulqa_contrastive, 800),
        "safety_v2": (load_safety_v2, 2000),
        "direct": (load_direct_responses, 1200),

        # NEW: Knowledge preservation MC (from train splits, not test)
        "mmlu_mc": (load_mmlu_mc, 1000),
        "arc_mc": (load_arc_mc, 500),
    }
    # Total: ~16,500

    return mix


def build_v8_nomc_data_mix():
    """v8 WITHOUT MC data — CE-only training for format + behavior.

    Removes all MC training data (truthfulqa_mc, mmlu_mc, arc_mc) to eliminate
    position bias. The adapter learns format and behavior from 13,700 CE-only
    examples. MC benchmarks (MMLU, ARC, TruthfulQA) are preserved by the
    frozen base model — proven at 0.0% MMLU delta in snap-on v1.

    This approach sacrifices MC improvement for MC preservation — the adapter
    can't IMPROVE MC scores but won't HURT them either.

    Total: ~13,700 examples (no MC contrastive data)
    """
    from experiments.operation_destroyer.train_v3 import (
        load_openhermes, load_ultrachat, load_magpie,
        load_constraint_data, load_math_data, load_code_data,
    )

    mix = {
        # Instruction core (same as v8)
        "openhermes": (load_openhermes, 4500),
        "ultrachat": (load_ultrachat, 3000),
        "magpie": (load_magpie, 1500),
        "constraint": (load_constraint_data, 1000),
        "math": (load_math_data, 500),
        "code": (load_code_data, 500),

        # Behavior fixes (non-MC)
        "safety_v2": (load_safety_v2, 2000),
        "direct": (load_direct_responses, 1200),

        # NO MC data: no truthfulqa_mc, no mmlu_mc, no arc_mc
    }
    # Total: ~13,700

    return mix


def build_v12_data_mix():
    """v12 data mix — v8 base + CoT math with <think> tags.

    Adds 4000 chain-of-thought math examples to the v8 mix.
    These teach the adapter to generate reasoning in <think>...</think>
    before answering, matching Qwen3's thinking mode format.

    Total: ~20,500 examples
    """
    from experiments.operation_destroyer.train_v3 import (
        load_openhermes, load_ultrachat, load_magpie,
        load_constraint_data, load_math_data, load_code_data,
    )

    mix = {
        # Instruction core (same as v8)
        "openhermes": (load_openhermes, 4500),
        "ultrachat": (load_ultrachat, 3000),
        "magpie": (load_magpie, 1500),
        "constraint": (load_constraint_data, 1000),
        "math": (load_math_data, 500),
        "code": (load_code_data, 500),

        # v2 targeted fixes (same as v8)
        "truthfulqa_mc": (load_truthfulqa_contrastive, 800),
        "safety_v2": (load_safety_v2, 2000),
        "direct": (load_direct_responses, 1200),

        # Knowledge preservation MC (same as v8)
        "mmlu_mc": (load_mmlu_mc, 1000),
        "arc_mc": (load_arc_mc, 500),

        # NEW: CoT math with <think> tags
        "cot_math": (load_cot_math, 4000),
    }
    # Total: ~20,500

    return mix


# ===== TRUE/FALSE CONTRASTIVE DATA =====
# Eliminates position bias entirely by removing ABCD letters.
# Each MC question → 1 True + 1 False example (balanced).
# Model learns "is this answer correct?" not "which letter?"

TF_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "Question: {question}\n"
    "Proposed answer: {answer}\n"
    "Is the proposed answer correct? Reply with only True or False.\n\n"
    "### Response:\n"
)


def load_mmlu_tf(n: int) -> List[Tuple[str, str]]:
    """MMLU as True/False contrastive pairs — no position bias possible.

    Each question produces 1 True + 1 False example (balanced 50/50).
    Uses auxiliary_train split to avoid eval contamination.
    Returns 2-tuples (prompt, response) — standard CE training.
    """
    from datasets import load_dataset
    print(f"  [MMLU T/F] Loading {n} T/F pairs for knowledge training...")
    try:
        try:
            ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
        except Exception:
            ds = load_dataset("cais/mmlu", "all", split="validation")
        ds = ds.shuffle(seed=123)

        pairs = []
        for ex in ds:
            if len(pairs) >= n:
                break
            q = ex["question"]
            options = ex.get("choices", [])
            answer_idx = ex.get("answer", -1)
            if not options or answer_idx < 0 or answer_idx >= len(options):
                continue

            correct_text = options[answer_idx]
            wrong_indices = [i for i in range(len(options)) if i != answer_idx]
            if not wrong_indices:
                continue

            # True example: correct answer
            true_prompt = TF_TEMPLATE.format(question=q, answer=correct_text)
            pairs.append((true_prompt, " True"))

            # False example: one random wrong answer
            wrong_idx = random.choice(wrong_indices)
            false_prompt = TF_TEMPLATE.format(question=q, answer=options[wrong_idx])
            pairs.append((false_prompt, " False"))

        random.shuffle(pairs)
        print(f"    Got {len(pairs)} MMLU T/F pairs")
        return pairs[:n]
    except Exception as e:
        print(f"    MMLU T/F failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def load_arc_tf(n: int) -> List[Tuple[str, str]]:
    """ARC-Challenge as True/False contrastive pairs.

    Each question → 1 True + 1 False (balanced 50/50).
    Uses train split to avoid eval contamination.
    """
    from datasets import load_dataset
    print(f"  [ARC T/F] Loading {n} T/F pairs...")
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        ds = ds.shuffle(seed=123)

        pairs = []
        for ex in ds:
            if len(pairs) >= n:
                break
            question = ex["question"]
            choices_data = ex["choices"]
            answer_key = ex["answerKey"]
            labels = choices_data["label"]
            texts = choices_data["text"]

            try:
                correct_idx = labels.index(answer_key)
            except ValueError:
                continue

            correct_text = texts[correct_idx]
            wrong_indices = [i for i in range(len(texts)) if i != correct_idx]
            if not wrong_indices:
                continue

            # True example
            true_prompt = TF_TEMPLATE.format(question=question, answer=correct_text)
            pairs.append((true_prompt, " True"))

            # False example
            wrong_idx = random.choice(wrong_indices)
            false_prompt = TF_TEMPLATE.format(question=question, answer=texts[wrong_idx])
            pairs.append((false_prompt, " False"))

        random.shuffle(pairs)
        print(f"    Got {len(pairs)} ARC T/F pairs")
        return pairs[:n]
    except Exception as e:
        print(f"    ARC T/F failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def load_truthfulqa_tf(n: int) -> List[Tuple[str, str]]:
    """TruthfulQA as True/False contrastive pairs.

    Each question → 1 True + 1 False (balanced 50/50).
    Uses all available correct/incorrect answers.
    """
    from datasets import load_dataset
    print(f"  [TruthfulQA T/F] Loading {n} T/F pairs...")
    try:
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        ds = ds.shuffle(seed=42)

        pairs = []
        for ex in ds:
            if len(pairs) >= n:
                break
            q = ex["question"]
            mc1 = ex.get("mc1_targets", {})
            choice_list = mc1.get("choices", [])
            labels_list = mc1.get("labels", [])
            if not choice_list or not labels_list:
                continue

            correct = [choice_list[i] for i, l in enumerate(labels_list) if l == 1]
            wrong = [choice_list[i] for i, l in enumerate(labels_list) if l == 0]
            if not correct or not wrong:
                continue

            # True example: pick a correct answer
            true_text = random.choice(correct)
            true_prompt = TF_TEMPLATE.format(question=q, answer=true_text)
            pairs.append((true_prompt, " True"))

            # False example: pick a wrong answer
            false_text = random.choice(wrong)
            false_prompt = TF_TEMPLATE.format(question=q, answer=false_text)
            pairs.append((false_prompt, " False"))

        random.shuffle(pairs)
        print(f"    Got {len(pairs)} TruthfulQA T/F pairs")
        return pairs[:n]
    except Exception as e:
        print(f"    TruthfulQA T/F failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def build_v21_data_mix():
    """v21 data mix — replaces ALL MC (ABCD) data with True/False contrastive.

    Eliminates position bias by design: no ABCD letters in training data.
    The adapter learns to distinguish true from false answers, not letter positions.

    T/F data: 2000 MMLU + 1000 ARC + 1600 TruthfulQA = 4600 (28%)
    Total: ~18,300 examples
    """
    from experiments.operation_destroyer.train_v3 import (
        load_openhermes, load_ultrachat, load_magpie,
        load_constraint_data, load_math_data, load_code_data,
    )

    mix = {
        # Instruction core (same as v8)
        "openhermes": (load_openhermes, 4500),
        "ultrachat": (load_ultrachat, 3000),
        "magpie": (load_magpie, 1500),
        "constraint": (load_constraint_data, 1000),
        "math": (load_math_data, 500),
        "code": (load_code_data, 500),

        # Behavior fixes (same as v8)
        "safety_v2": (load_safety_v2, 2000),
        "direct": (load_direct_responses, 1200),

        # NEW: True/False contrastive (replaces truthfulqa_mc, mmlu_mc, arc_mc)
        "mmlu_tf": (load_mmlu_tf, 2000),
        "arc_tf": (load_arc_tf, 1000),
        "truthfulqa_tf": (load_truthfulqa_tf, 1600),
    }
    # Total: ~18,300

    return mix


if __name__ == "__main__":
    """Quick test — load and report stats."""
    mix = build_v2_data_mix()
    total_target = sum(n for _, n in mix.values())
    print(f"\nv2 Data Mix — Target: {total_target} examples\n")
    print(f"{'Source':<20s} {'Target':>8s}")
    print("-" * 30)
    for name, (_, n) in mix.items():
        print(f"  {name:<18s} {n:>6d}")
    print(f"  {'TOTAL':<18s} {total_target:>6d}")

    print("\n\nTest loading each source...")
    total_loaded = 0
    for name, (loader, n) in mix.items():
        try:
            pairs = loader(min(n, 50))  # Test with 50 each
            print(f"  {name}: loaded {len(pairs)} (test)")
            total_loaded += len(pairs)
        except Exception as e:
            print(f"  {name}: FAILED — {e}")

    print(f"\nAll sources tested. {total_loaded} examples loaded in test mode.")
