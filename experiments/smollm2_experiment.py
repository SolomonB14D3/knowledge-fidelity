#!/usr/bin/env python3
"""SmolLM2-360M Pipeline Experiment.

Tests the full rho-unlock → snap-on pipeline on HuggingFaceTB/SmolLM2-360M,
a very small model where expression gaps should be large.

Pipeline:
  1. rho-unlock diagnose SmolLM2-360M base (measure expression gaps)
  2. CD rescue with SmolLM2-135M as amateur
  3. Train logit adapter (very fast on 360M)
  4. Compare against SmolLM2-360M-Instruct
  5. Qualitative conversation tests

Usage:
    python experiments/smollm2_experiment.py
"""

import json
import os
import sys
import time
import traceback

import mlx.core as mx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

RESULTS_DIR = os.path.join(ROOT, "results", "smollm2_360m")

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M"
AMATEUR_MODEL = "HuggingFaceTB/SmolLM2-135M"
INSTRUCT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


def step_1_diagnose_base():
    """Diagnose expression gaps on SmolLM2-360M base."""
    out_file = os.path.join(RESULTS_DIR, "diagnose.json")
    if os.path.exists(out_file):
        print(f"[Step 1] Diagnose already exists at {out_file}")
        with open(out_file) as f:
            return json.load(f)

    print("=" * 70)
    print("STEP 1: Diagnose SmolLM2-360M Base — Expression Gaps")
    print("=" * 70)

    from rho_eval.cli.rho_unlock import main as rho_unlock_main

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rho_unlock_main([
        "diagnose", BASE_MODEL,
        "--behaviors", "bias,sycophancy,mmlu,truthfulqa,arc,hellaswag",
        "--output", out_file,
    ])

    if os.path.exists(out_file):
        with open(out_file) as f:
            results = json.load(f)
        print(f"\nExpression gaps:")
        for beh, gap in results.get("expression_gaps", {}).items():
            diag = results.get("diagnoses", {}).get(beh, {})
            quad = diag.get("quadrant", "?")
            print(f"  {beh:15s}: gap={gap:+.1%}  quadrant={quad}")
        return results
    return None


def step_2_cd_rescue():
    """CD rescue with SmolLM2-135M as amateur."""
    out_file = os.path.join(RESULTS_DIR, "unlock.json")
    if os.path.exists(out_file):
        print(f"[Step 2] CD unlock already exists at {out_file}")
        with open(out_file) as f:
            return json.load(f)

    print("\n" + "=" * 70)
    print("STEP 2: Contrastive Decoding Rescue (amateur: SmolLM2-135M)")
    print("=" * 70)

    # Check which behaviors need unlocking
    diagnose_file = os.path.join(RESULTS_DIR, "diagnose.json")
    behaviors_to_unlock = ["mmlu", "truthfulqa", "arc", "hellaswag"]

    if os.path.exists(diagnose_file):
        with open(diagnose_file) as f:
            diag = json.load(f)
        # Only unlock behaviors that have meaningful gaps
        unlock_behaviors = []
        for beh in behaviors_to_unlock:
            gap = diag.get("expression_gaps", {}).get(beh, 0)
            if gap > 0.03:  # Only unlock if gap > 3%
                unlock_behaviors.append(beh)
        if not unlock_behaviors:
            print("  No behaviors with meaningful expression gaps. Skipping CD.")
            # Save empty result
            os.makedirs(RESULTS_DIR, exist_ok=True)
            with open(out_file, "w") as f:
                json.dump({"skipped": True, "reason": "no_gaps"}, f, indent=2)
            return {"skipped": True}
        behaviors_to_unlock = unlock_behaviors

    from rho_eval.cli.rho_unlock import main as rho_unlock_main

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Set amateur model for SmolLM2
    # Need to check if AMATEUR_MAP has this model, otherwise pass explicitly
    rho_unlock_main([
        "unlock", BASE_MODEL,
        "--behaviors", ",".join(behaviors_to_unlock),
        "--alpha", "0.5",
        "--amateur", AMATEUR_MODEL,
        "--output", out_file,
    ])

    if os.path.exists(out_file):
        with open(out_file) as f:
            return json.load(f)
    return None


def step_3_train_adapter():
    """Train logit adapter on SmolLM2-360M (should be very fast)."""
    from mlx.utils import tree_flatten
    import mlx_lm

    from rho_eval.snap_on import (
        SnapOnConfig, create_adapter,
        load_alpaca_data, train, save_adapter,
        evaluate_mmlu,
    )

    save_dir = os.path.join(RESULTS_DIR, "adapter")
    results_file = os.path.join(save_dir, "results.json")

    if os.path.exists(results_file):
        print(f"[Step 3] Adapter training results already exist at {results_file}")
        with open(results_file) as f:
            return json.load(f)

    print("\n" + "=" * 70)
    print("STEP 3: Train Logit Adapter on SmolLM2-360M")
    print("=" * 70)

    print(f"\nLoading base model: {BASE_MODEL}")
    base_model, tokenizer = mlx_lm.load(BASE_MODEL)
    base_model.freeze()

    d_model = base_model.model.layers[0].self_attn.q_proj.weight.shape[0]
    if hasattr(base_model, "args") and hasattr(base_model.args, "vocab_size"):
        vocab_size = base_model.args.vocab_size
    else:
        vocab_size = base_model.model.embed_tokens.weight.shape[0]
    print(f"  d_model = {d_model}, vocab_size = {vocab_size}")

    # Load data
    train_examples, val_examples = load_alpaca_data(
        tokenizer, n_train=10000, n_val=500, max_seq_len=512
    )

    # Create adapter
    config = SnapOnConfig(
        d_model=d_model,
        d_inner=64,
        mode="logit",
        vocab_size=vocab_size,
    )
    adapter = create_adapter(config)

    n_params = sum(p.size for _, p in tree_flatten(adapter.parameters()))
    print(f"\nAdapter: mode=logit, d_inner=64")
    print(f"  Total params: {n_params:,} ({n_params * 4 / 1e6:.1f} MB)")

    t0 = time.time()
    best_val = train(
        base_model, tokenizer, adapter,
        train_examples, val_examples,
        epochs=3,
        lr=1e-5,
        warmup_steps=100,
        log_every=100,
        eval_every=1000,
        save_dir=save_dir,
        mode="logit",
    )
    train_time = time.time() - t0

    # MMLU evaluation
    print("\nEvaluating MMLU...")
    base_acc, adapter_acc = evaluate_mmlu(
        base_model, adapter, tokenizer, n_questions=500, mode="logit"
    )

    results = {
        "model": BASE_MODEL,
        "mode": "logit",
        "d_model": d_model,
        "vocab_size": vocab_size,
        "d_inner": 64,
        "n_params": n_params,
        "n_train": 10000,
        "epochs": 3,
        "best_val_loss": best_val,
        "train_time_s": train_time,
        "mmlu_base_acc": base_acc,
        "mmlu_adapter_acc": adapter_acc,
        "mmlu_delta": adapter_acc - base_acc if base_acc else None,
    }

    os.makedirs(save_dir, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    print(f"  Training time: {train_time:.0f}s ({train_time/60:.1f} min)")
    return results


def step_4_diagnose_instruct():
    """Diagnose SmolLM2-360M-Instruct for comparison."""
    out_file = os.path.join(RESULTS_DIR, "diagnose_instruct.json")
    if os.path.exists(out_file):
        print(f"[Step 4] Instruct diagnose already exists at {out_file}")
        with open(out_file) as f:
            return json.load(f)

    print("\n" + "=" * 70)
    print("STEP 4: Diagnose SmolLM2-360M-Instruct (comparison)")
    print("=" * 70)

    from rho_eval.cli.rho_unlock import main as rho_unlock_main

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rho_unlock_main([
        "diagnose", INSTRUCT_MODEL,
        "--behaviors", "bias,sycophancy,mmlu,truthfulqa,arc,hellaswag",
        "--output", out_file,
    ])

    if os.path.exists(out_file):
        with open(out_file) as f:
            results = json.load(f)
        print(f"\nInstruct expression gaps:")
        for beh, gap in results.get("expression_gaps", {}).items():
            print(f"  {beh:15s}: gap={gap:+.1%}")
        return results
    return None


def step_5_qualitative():
    """Qualitative conversation tests."""
    from rho_eval.snap_on import (
        load_adapter, ALPACA_TEMPLATE,
        generate_with_adapter, generate_base_only,
    )
    import mlx_lm

    out_file = os.path.join(RESULTS_DIR, "qualitative.json")
    if os.path.exists(out_file):
        print(f"[Step 5] Qualitative already exists at {out_file}")
        return

    print("\n" + "=" * 70)
    print("STEP 5: Qualitative Conversation Tests")
    print("=" * 70)

    # Load models
    base_model, tokenizer = mlx_lm.load(BASE_MODEL)
    base_model.freeze()

    adapter_dir = os.path.join(RESULTS_DIR, "adapter")
    adapter = load_adapter(adapter_dir, "best")

    instruct_model, instruct_tokenizer = mlx_lm.load(INSTRUCT_MODEL)
    instruct_model.freeze()

    test_prompts = [
        "What is the capital of France?",
        "Explain what a neural network is in simple terms.",
        "Write a Python function to calculate factorial.",
        "What causes thunder and lightning?",
        "List 3 healthy breakfast options.",
        "What is the difference between a planet and a star?",
        "How does the internet work?",
        "What is machine learning?",
        "Why is the sky blue?",
        "Write a short story about a robot.",
    ]

    results = []
    for prompt_text in test_prompts:
        full_prompt = ALPACA_TEMPLATE.format(instruction=prompt_text)
        print(f"\n{'─' * 60}")
        print(f"INSTRUCTION: {prompt_text}")

        # Base model
        base_out = generate_base_only(base_model, tokenizer, full_prompt, max_tokens=150)
        print(f"\n[BASE]: {base_out[:200]}")

        # Adapter
        adapter_out = generate_with_adapter(
            base_model, adapter, tokenizer, full_prompt,
            max_tokens=150, mode="logit"
        )
        print(f"\n[ADAPTER]: {adapter_out[:200]}")

        # Instruct
        instruct_msgs = [{"role": "user", "content": prompt_text}]
        try:
            instruct_prompt = instruct_tokenizer.apply_chat_template(
                instruct_msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            instruct_prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        instruct_out = generate_base_only(
            instruct_model, instruct_tokenizer, instruct_prompt, max_tokens=150
        )
        print(f"\n[INSTRUCT]: {instruct_out[:200]}")

        results.append({
            "prompt": prompt_text,
            "base": base_out,
            "adapter": adapter_out,
            "instruct": instruct_out,
        })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")


def step_6_summary():
    """Print final summary comparing all results."""
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: SmolLM2-360M Pipeline")
    print("=" * 70)

    # Diagnose
    diag_file = os.path.join(RESULTS_DIR, "diagnose.json")
    if os.path.exists(diag_file):
        with open(diag_file) as f:
            diag = json.load(f)
        print(f"\n1. Base model expression gaps:")
        for beh, gap in diag.get("expression_gaps", {}).items():
            d = diag.get("diagnoses", {}).get(beh, {})
            print(f"   {beh:15s}: gap={gap:+.1%}  "
                  f"logit={d.get('logit_accuracy', 'N/A'):.1%}  "
                  f"gen={d.get('gen_accuracy', 'N/A'):.1%}  "
                  f"quadrant={d.get('quadrant', '?')}")

    # CD rescue
    unlock_file = os.path.join(RESULTS_DIR, "unlock.json")
    if os.path.exists(unlock_file):
        with open(unlock_file) as f:
            unlock = json.load(f)
        if not unlock.get("skipped"):
            print(f"\n2. Contrastive decoding rescue (alpha=0.5):")
            for beh, data in unlock.get("results", {}).items():
                if isinstance(data, dict):
                    base = data.get("base_accuracy", data.get("base_acc", "?"))
                    cd = data.get("cd_accuracy", data.get("cd_acc", "?"))
                    print(f"   {beh:15s}: base={base}, cd={cd}")
        else:
            print(f"\n2. CD rescue: skipped (no meaningful gaps)")

    # Adapter
    adapter_file = os.path.join(RESULTS_DIR, "adapter", "results.json")
    if os.path.exists(adapter_file):
        with open(adapter_file) as f:
            adapter = json.load(f)
        print(f"\n3. Logit adapter MMLU:")
        print(f"   Base:    {adapter.get('mmlu_base_acc', 'N/A'):.1%}")
        print(f"   Adapter: {adapter.get('mmlu_adapter_acc', 'N/A'):.1%}")
        print(f"   Delta:   {adapter.get('mmlu_delta', 'N/A'):+.1%}")
        print(f"   Training time: {adapter.get('train_time_s', 0)/60:.1f} min")

    # Instruct comparison
    instruct_file = os.path.join(RESULTS_DIR, "diagnose_instruct.json")
    if os.path.exists(instruct_file):
        with open(instruct_file) as f:
            instruct = json.load(f)
        print(f"\n4. SmolLM2-360M-Instruct comparison:")
        for beh in ["mmlu", "arc", "truthfulqa", "hellaswag"]:
            gap = instruct.get("expression_gaps", {}).get(beh, "N/A")
            gen = instruct.get("gen_accuracies", {}).get(beh, "N/A")
            if isinstance(gen, float):
                print(f"   {beh:15s}: gen_acc={gen:.1%}, gap={gap:+.1%}")

    print("\n" + "=" * 70)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    steps = [
        ("Step 1: Diagnose base model", step_1_diagnose_base),
        ("Step 2: CD rescue", step_2_cd_rescue),
        ("Step 3: Train logit adapter", step_3_train_adapter),
        ("Step 4: Diagnose instruct model", step_4_diagnose_instruct),
        ("Step 5: Qualitative tests", step_5_qualitative),
        ("Step 6: Final summary", step_6_summary),
    ]

    for name, fn in steps:
        print(f"\n{'#' * 70}")
        print(f"# {name}")
        print(f"{'#' * 70}\n")
        try:
            fn()
        except Exception as e:
            print(f"\n*** {name} FAILED: {e}")
            traceback.print_exc()
            continue

    print("\nSmolLM2-360M experiment complete!")


if __name__ == "__main__":
    main()
