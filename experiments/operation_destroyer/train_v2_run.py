#!/usr/bin/env python3
"""Operation Destroyer v2 — retrain with improved data mix.

Uses v2 data pipeline (MC knowledge, contrastive TruthfulQA, diverse refusals)
and trains for exactly 1 epoch. Evaluates on all benchmarks at the end.
"""

import argparse
import json
import os
import sys
import time
import random

sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity")
sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/snap_on")

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_lm
from mlx.utils import tree_flatten

from module import SnapOnConfig, create_adapter
from experiments.operation_destroyer.train_v3 import (
    get_lm_head_fn, _tokenize_pairs, save_adapter_only, save_checkpoint,
    train, evaluate_mmlu, evaluate_truthfulqa, evaluate_arc,
    evaluate_safety, evaluate_qualitative,
    RESULTS_DIR, MAX_SEQ_LEN, N_VAL,
)
from experiments.operation_destroyer.train_v3_data_v2 import build_v2_data_mix, build_v4_data_mix, build_v8_data_mix, build_v8_nomc_data_mix, build_v12_data_mix, build_v21_data_mix


def load_v2_data(tokenizer, max_seq_len, val_ratio=0.02, data_mix_name="v2"):
    """Load data sources and split into train/val."""
    if data_mix_name == "v21":
        mix = build_v21_data_mix()
    elif data_mix_name == "v12":
        mix = build_v12_data_mix()
    elif data_mix_name == "v8_nomc":
        mix = build_v8_nomc_data_mix()
    elif data_mix_name == "v8":
        mix = build_v8_data_mix()
    elif data_mix_name == "v4":
        mix = build_v4_data_mix()
    else:
        mix = build_v2_data_mix()
    all_train = []
    all_val = []
    source_stats = {}

    for name, (loader, n_target) in mix.items():
        print(f"\n{'=' * 60}")
        print(f"  Loading: {name} (target: {n_target})")
        print(f"{'=' * 60}")

        t0 = time.time()
        try:
            pairs = loader(n_target)
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")
            import traceback
            traceback.print_exc()
            pairs = []

        if not pairs:
            print(f"  WARNING: No data from {name}")
            source_stats[name] = {"loaded": 0, "tokenized": 0}
            continue

        examples, skipped = _tokenize_pairs(pairs, tokenizer, max_seq_len)
        elapsed = time.time() - t0

        n_val = max(int(len(examples) * val_ratio), 1)
        n_val = min(n_val, len(examples) // 5)
        random.shuffle(examples)
        val_portion = examples[:n_val]
        train_portion = examples[n_val:]

        all_train.extend(train_portion)
        all_val.extend(val_portion)

        avg_len = sum(len(e["tokens"]) for e in examples) / max(len(examples), 1)
        source_stats[name] = {
            "loaded": len(pairs),
            "tokenized": len(examples),
            "skipped": skipped,
            "train": len(train_portion),
            "val": len(val_portion),
            "avg_tokens": avg_len,
            "time_s": elapsed,
        }
        print(f"  {name}: {len(pairs)} loaded -> {len(examples)} tokenized "
              f"({skipped} skipped), avg {avg_len:.0f} tokens, {elapsed:.1f}s")

    random.seed(42)
    random.shuffle(all_train)
    random.shuffle(all_val)

    if len(all_val) > N_VAL:
        all_val = all_val[:N_VAL]

    print(f"\n{'=' * 60}")
    print(f"  V2 DATA SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total train: {len(all_train)}")
    print(f"  Total val:   {len(all_val)}")
    for name, stats in source_stats.items():
        print(f"    {name:>20s}: {stats.get('train', 0):>6d} train, "
              f"{stats.get('val', 0):>4d} val, "
              f"avg {stats.get('avg_tokens', 0):.0f} tok")

    return all_train, all_val, source_stats


def main():
    parser = argparse.ArgumentParser(description="Operation Destroyer v2")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--d_inner", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=None)  # Auto: 1 for v2/v3, 3 for v4
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--mmlu_n", type=int, default=500)
    parser.add_argument("--truthfulqa_n", type=int, default=200)
    parser.add_argument("--arc_n", type=int, default=200)
    parser.add_argument("--safety_n", type=int, default=25)
    parser.add_argument("--version", default="v4", help="Version tag for save dir")
    parser.add_argument("--data_mix", default=None, help="Data mix: v2 or v4 (auto from version)")
    parser.add_argument("--max_shift", type=float, default=None,
                        help="Override ADAPTER_MAX_SHIFT in train_v3")
    parser.add_argument("--progressive_cap", action="store_true",
                        help="Progressive cap: start at 0.3, expand to max_shift after epoch 1")
    parser.add_argument("--softcap", type=float, default=30.0,
                        help="Logit softcap (Gemma 2 style, default: 30.0)")
    parser.add_argument("--pos_bias_weight", type=float, default=0.0,
                        help="Position-bias penalty weight (0=disabled, 0.05=weak, 0.3=strong)")
    parser.add_argument("--pos_bias_mode", default="wrong_only",
                        choices=["all", "wrong_only"],
                        help="Position bias mode: 'all' penalizes ABCD variance, "
                             "'wrong_only' penalizes only wrong-answer variance")
    parser.add_argument("--elimination", action="store_true",
                        help="v19+: Elimination training — suppress 2 obviously wrong MC answers")
    parser.add_argument("--kl_teacher", type=str, default=None,
                        help="Teacher model for KL distillation (e.g. 'Qwen/Qwen3-4B')")
    parser.add_argument("--kl_alpha", type=float, default=0.7,
                        help="KL blend: alpha*KL + (1-alpha)*CE (default 0.7)")
    parser.add_argument("--kl_temperature", type=float, default=2.0,
                        help="Distillation temperature (default 2.0)")
    args = parser.parse_args()

    # Override globals in train_v3
    import experiments.operation_destroyer.train_v3 as t3
    if args.max_shift is not None:
        t3.ADAPTER_MAX_SHIFT = args.max_shift
        print(f"  [Override] ADAPTER_MAX_SHIFT = {args.max_shift}")
    t3.LOGIT_SOFTCAP = args.softcap
    print(f"  [Softcap] LOGIT_SOFTCAP = {args.softcap}")
    t3.POSITION_BIAS_WEIGHT = args.pos_bias_weight
    t3.POSITION_BIAS_MODE = args.pos_bias_mode
    if args.pos_bias_weight > 0:
        print(f"  [Position bias] weight={args.pos_bias_weight}, mode={args.pos_bias_mode}")
    if args.elimination:
        t3.ELIMINATION_MODE = True
        print(f"  [Elimination mode] Training to suppress 2 obviously wrong MC answers")
    if args.progressive_cap:
        # Start at 0.3, will expand to max_shift after epoch 1
        t3.ADAPTER_MAX_SHIFT = 0.3
        print(f"  [Progressive cap] Starting at 0.3, expanding to {args.max_shift} after epoch 1")

    save_dir = os.path.join(RESULTS_DIR, args.version)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("  OPERATION DESTROYER v2")
    print("  Improved data mix: MC knowledge + contrastive TruthfulQA + diverse refusals")
    print("=" * 70)

    # Load model
    print(f"\nLoading {args.model}...")
    model, tokenizer = mlx_lm.load(args.model)
    model.freeze()
    lm_head = get_lm_head_fn(model)
    try:
        vocab_size = lm_head.weight.shape[0]
    except AttributeError:
        vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    print(f"  d_model={d_model}, vocab_size={vocab_size}")

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
        _run_eval(model, tokenizer, adapter, lm_head, args, save_dir)
        return

    # Load v2 data
    print(f"\n{'#' * 70}")
    print(f"#  LOADING V2 DATA")
    print(f"{'#' * 70}")

    # Auto-detect data mix from version tag if not explicit
    if args.data_mix:
        data_mix_name = args.data_mix
    elif args.version.startswith("v21"):
        data_mix_name = "v21"
    elif args.version in ("v12",):
        data_mix_name = "v12"
    elif args.version in ("v8", "v9"):
        data_mix_name = "v8"
    elif args.version in ("v4", "v5", "v6", "v7"):
        data_mix_name = "v4"
    else:
        data_mix_name = "v2"
    epochs = args.epochs or (3 if data_mix_name in ("v4", "v8") else 1)

    train_examples, val_examples, source_stats = load_v2_data(
        tokenizer, args.max_seq_len, data_mix_name=data_mix_name
    )

    if args.smoke_test:
        train_examples = train_examples[:200]
        val_examples = val_examples[:50]
        epochs = 1  # Override: smoke tests are always 1 epoch
        print(f"\n*** SMOKE TEST: {len(train_examples)} train, {len(val_examples)} val, 1 epoch ***")

    with open(os.path.join(save_dir, "data_stats.json"), "w") as f:
        json.dump(source_stats, f, indent=2, default=str)

    # Train (1 epoch by default)
    print(f"\n{'#' * 70}")
    print(f"#  TRAINING ({data_mix_name} data, {epochs} epoch{'s' if epochs > 1 else ''})")
    print(f"{'#' * 70}")

    # Warmup: use ratio (passed as float < 1) or fixed count
    warmup = args.warmup_ratio  # 0.05 = 5% of total steps

    # Log every 200 steps, eval every 1000 steps (but more frequent for small datasets)
    n_train = len(train_examples)
    log_every = min(200, max(n_train // 50, 50))
    eval_every = min(1000, max(n_train // 5, 200))

    # Wire progressive cap: pass target to train() so it expands after epoch 1
    prog_cap = args.max_shift if args.progressive_cap else None

    # KL distillation: load teacher model if specified
    teacher_model = None
    if args.kl_teacher:
        print(f"\nLoading teacher model: {args.kl_teacher}")
        teacher_model, _ = mlx_lm.load(args.kl_teacher)
        teacher_model.freeze()
        print(f"  Teacher loaded (frozen)")
        t3.KL_ALPHA = args.kl_alpha
        t3.KL_TEMPERATURE = args.kl_temperature

    t0 = time.time()
    best_val_loss, avg_base_loss, training_log = train(
        model, tokenizer, adapter,
        train_examples, val_examples,
        epochs=epochs, lr=args.lr,
        warmup_steps=warmup, log_every=log_every, eval_every=eval_every,
        save_dir=save_dir,
        progressive_cap_target=prog_cap,
        teacher_model=teacher_model,
        kl_alpha=args.kl_alpha,
        kl_temperature=args.kl_temperature,
    )
    train_time = time.time() - t0
    print(f"\n  Training time: {train_time/3600:.1f}h")

    # Load best weights
    best_path = os.path.join(save_dir, "best.npz")
    if os.path.exists(best_path):
        weights = mx.load(best_path)
        adapter.load_weights(list(weights.items()))
        mx.eval(adapter.parameters())

    if args.smoke_test:
        # Smoke test: run quick diagnostic instead of full eval
        print(f"\n{'#' * 70}")
        print(f"#  SMOKE TEST DIAGNOSTIC")
        print(f"{'#' * 70}")
        from experiments.operation_destroyer.train_v3 import _quick_mmlu_spot_check
        result = _quick_mmlu_spot_check(
            model, tokenizer, adapter, lm_head, n=50, save_dir=save_dir
        )
        if result:
            status = result["status"]
            if status != "OK":
                print(f"\n  *** SMOKE TEST FAILED: {status} ***")
                print(f"  Do NOT proceed to full run. Investigate first.")
            else:
                print(f"\n  SMOKE TEST PASSED. Safe to proceed to full run.")
            # Save smoke test result
            import json as _json
            with open(os.path.join(save_dir, "smoke_result.json"), "w") as f:
                _json.dump(result, f, indent=2)
    elif not args.skip_eval:
        _run_eval(model, tokenizer, adapter, lm_head, args, save_dir,
                  best_val_loss=best_val_loss, avg_base_loss=avg_base_loss,
                  train_time=train_time, source_stats=source_stats)


def _run_eval(model, tokenizer, adapter, lm_head, args, save_dir,
              best_val_loss=None, avg_base_loss=None,
              train_time=None, source_stats=None):
    """Run all benchmarks."""
    import traceback

    print(f"\n{'#' * 70}")
    print(f"#  EVALUATION")
    print(f"{'#' * 70}")

    results = {
        "version": "v2",
        "model": args.model,
        "d_inner": args.d_inner,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if best_val_loss: results["best_val_loss"] = best_val_loss
    if avg_base_loss: results["base_val_loss"] = avg_base_loss
    if train_time: results["train_time_hours"] = train_time / 3600
    if source_stats: results["data_sources"] = source_stats

    n_params = sum(p.size for _, p in tree_flatten(adapter.parameters()))
    results["n_params"] = n_params

    for bench_name, bench_fn, bench_kwargs in [
        ("mmlu", evaluate_mmlu, {"n_questions": args.mmlu_n}),
        ("truthfulqa_mc1", evaluate_truthfulqa, {"n_questions": args.truthfulqa_n}),
        ("arc_challenge", evaluate_arc, {"n_questions": args.arc_n}),
    ]:
        try:
            r = bench_fn(model, tokenizer, adapter, lm_head, **bench_kwargs)
            results[bench_name] = r
        except Exception as e:
            print(f"  {bench_name} failed: {e}")
            traceback.print_exc()
            results[bench_name] = {"error": str(e)}
        mx.clear_cache()

    try:
        safety = evaluate_safety(model, tokenizer, adapter, n_prompts=args.safety_n)
        results["safety"] = {
            "base_refusal_rate": safety["base_refusal_rate"],
            "adapter_refusal_rate": safety["adapter_refusal_rate"],
            "delta": safety["delta"],
        }
        with open(os.path.join(save_dir, "safety_details.json"), "w") as f:
            json.dump(safety["details"], f, indent=2)
    except Exception as e:
        print(f"  Safety failed: {e}")
        results["safety"] = {"error": str(e)}

    mx.clear_cache()

    try:
        qual = evaluate_qualitative(model, tokenizer, adapter)
        results["qualitative"] = qual
    except Exception as e:
        results["qualitative"] = {"error": str(e)}

    # Save results
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print comparison
    print(f"\n{'=' * 70}")
    print(f"  v2 RESULTS SUMMARY (compare to v1)")
    print(f"{'=' * 70}")

    v1_scores = {"mmlu": 66.2, "arc": 84.5, "truthfulqa": 32.0, "safety": 88.0}

    for key, label in [("mmlu", "MMLU"), ("truthfulqa_mc1", "TruthfulQA"),
                        ("arc_challenge", "ARC-Challenge")]:
        if key in results and isinstance(results[key], dict) and "adapter" in results[key]:
            v = results[key]
            v1 = v1_scores.get(key.split("_")[0], 0)
            print(f"  {label:15s}: base={v['base']:.1%} v2={v['adapter']:.1%} "
                  f"delta={v['delta']:+.1%}  (v1 was {v1:.1f}%)")

    if "safety" in results and "adapter_refusal_rate" in results["safety"]:
        s = results["safety"]
        print(f"  {'Safety':15s}: base={s['base_refusal_rate']:.1%} "
              f"v2={s['adapter_refusal_rate']:.1%} "
              f"delta={s['delta']:+.1%}  (v1 was {v1_scores['safety']:.1f}%)")

    print(f"\n  Results saved to {save_dir}/results.json")


if __name__ == "__main__":
    main()
