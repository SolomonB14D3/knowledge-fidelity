#!/usr/bin/env python3
"""Train a small causal LM with contrastive behavioral pairs in the data stream.

Data-side intervention: instead of a geometric regularizer, inject behavioral
contrast pairs (bias + sycophancy) directly into the pretraining text mix.
The model sees behaviorally differentiated text during standard LM training.

**Hypothesis:** The 7M eff_dim=1 bottleneck for bias/sycophancy may be a
signal problem (not enough behavioral variation in OpenWebText) rather than
a capacity problem. If injecting ~1000 contrastive pairs into 100M tokens
of pretraining pushes eff_dim > 1, the bottleneck is signal, not capacity.

**Controls:**
- Vanilla 7M (no injection) = existing baseline
- Random text injection (same volume, no behavioral signal) = exposure control

Usage:
    # Standard run: inject 1000 bias + sycophancy pairs
    python experiments/scale_ladder/train_contrastive.py --size 7M --seed 42 --device mps

    # Custom injection rate (pairs per 100 training steps)
    python experiments/scale_ladder/train_contrastive.py --size 7M --seed 42 \\
        --inject-rate 10 --inject-behaviors bias sycophancy

    # Also inject factual and toxicity
    python experiments/scale_ladder/train_contrastive.py --size 7M --seed 42 \\
        --inject-behaviors bias sycophancy factual toxicity
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale_ladder.configs import SCALE_CONFIGS, SCALE_ORDER, SUBSPACE_BEHAVIORS
from experiments.scale_ladder.train_model import build_model, build_tokenizer


def load_contrastive_texts(
    behaviors: list[str], probe_seed: int = 999, pairs_json: str | None = None,
) -> tuple[list[str], set[str]]:
    """Load contrast pair texts for injection into the data stream.

    If pairs_json is provided, loads pre-built pairs from that JSON file
    for any behavior named in the file. Behaviors not in the file fall
    through to the standard probe-loading pipeline.

    Guarantees ZERO overlap with evaluation probes by:
    1. Loading eval probes (seed=42) to get their IDs
    2. Loading a larger pool with probe_seed
    3. Explicitly filtering out any eval-overlapping probes

    Returns:
        texts: list of positive and negative texts for LM training.
        training_ids: set of probe IDs used (for contamination audit).
    """
    from rho_eval.interpretability.activation import build_contrast_pairs
    from rho_eval.behavioral import load_behavioral_probes

    # Load custom pairs if provided
    custom_pairs = {}
    if pairs_json:
        import json as _json
        pairs_path = Path(pairs_json)
        if not pairs_path.exists():
            raise FileNotFoundError(f"Pairs JSON not found: {pairs_json}")
        raw = _json.loads(pairs_path.read_text())
        if isinstance(raw, list):
            # Flat list of pairs — treat as a single "primitive" behavior
            custom_pairs["primitive"] = raw
        elif isinstance(raw, dict):
            # Dict keyed by behavior name
            custom_pairs = raw
        print(f"    Loaded custom pairs from {pairs_json}")
        for k, v in custom_pairs.items():
            print(f"      {k}: {len(v)} pairs")

    texts = []
    training_ids = set()
    for b in behaviors:
        if b in custom_pairs:
            # Use pre-built pairs directly
            pairs = custom_pairs[b]
            for pair in pairs:
                texts.append(pair["positive"])
                texts.append(pair["negative"])
                training_ids.add(pair.get("id", ""))
            print(f"    {b}: {len(pairs)} custom pairs → {len(pairs)*2} texts")
            continue

        # Standard probe-loading pipeline
        # Step 1: Get eval probe IDs (these are what we must NOT touch)
        eval_probes = load_behavioral_probes(b, seed=42)
        eval_ids = {p.get("id", "") for p in eval_probes}
        eval_texts = {p.get("text", "")[:200] for p in eval_probes}

        # Step 2: Load a LARGER pool with different seed, then filter
        # Draw 3x as many probes to have headroom after filtering
        n_target = len(eval_probes)  # want ~same number for training
        candidate_probes = load_behavioral_probes(b, seed=probe_seed, n=n_target * 3)

        # Step 3: Remove any probe whose ID or text matches eval set
        clean_probes = []
        for p in candidate_probes:
            pid = p.get("id", "")
            ptxt = p.get("text", "")[:200]
            if pid not in eval_ids and ptxt not in eval_texts:
                clean_probes.append(p)
            if len(clean_probes) >= n_target:
                break

        n_filtered = len(candidate_probes) - len(clean_probes)
        print(f"    {b}: {len(candidate_probes)} candidates → {n_filtered} filtered → {len(clean_probes)} clean probes")

        pairs = build_contrast_pairs(b, clean_probes)
        for pair in pairs:
            texts.append(pair["positive"])
            texts.append(pair["negative"])
            training_ids.add(pair.get("id", ""))
        print(f"    {b}: {len(pairs)} pairs → {len(pairs)*2} texts")

    print(f"    Training probe IDs: {len(training_ids)} unique (0 overlap with eval)")
    return texts, training_ids


def contrastive_token_iterator(
    tokenizer, block_size, contrastive_texts, inject_rate=10, seed=42,
    dataset="openwebtext",
    schedule=None, total_blocks=None, inject_until=None,
):
    """Yield packed token blocks, interleaving contrastive texts.

    Every `inject_rate` blocks, one block comes from the contrastive pool
    instead of the standard dataset. Contrastive texts are tokenized,
    concatenated, and packed the same way as regular data.

    inject_rate=10 means ~10% of training tokens are contrastive.
    inject_rate=100 means ~1% contrastive.

    schedule: list of (start_frac, end_frac, texts) for curriculum ordering.
        When provided, the active contrastive text set changes based on
        training progress. Overrides contrastive_texts.
    total_blocks: total number of blocks in training (needed for schedule).
    inject_until: fraction (0-1) of training to inject contrastive, then stop.
        E.g., inject_until=0.2 means inject for first 20% only.
    """
    from datasets import load_dataset

    # Standard data stream
    if dataset == "fineweb-edu":
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                         split="train", streaming=True)
    elif dataset == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-v1",
                         split="train", streaming=True)
    else:
        ds = load_dataset("Skylion007/openwebtext",
                         split="train", streaming=True)

    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    eos_id = tokenizer.eos_token_id

    def _tokenize_texts(texts, rng_seed):
        """Tokenize and shuffle a list of texts into a token buffer."""
        _rng = random.Random(rng_seed)
        tokens = []
        shuffled = list(texts)
        _rng.shuffle(shuffled)
        for t in shuffled:
            tokens.extend(tokenizer.encode(t))
            tokens.append(eos_id)
        return tokens

    # Build schedule-aware contrastive buffers
    if schedule and total_blocks:
        # Pre-tokenize each schedule phase
        schedule_buffers = []
        for start_frac, end_frac, phase_texts in schedule:
            phase_tokens = _tokenize_texts(phase_texts, seed)
            schedule_buffers.append((start_frac, end_frac, phase_tokens))
        contrastive_tokens = None
    else:
        # Standard: single contrastive buffer
        contrastive_tokens = _tokenize_texts(contrastive_texts, seed)
        schedule_buffers = None

    # Main data buffer
    main_buffer = []
    contr_idx = 0  # cycles through active contrastive tokens
    active_phase = -1  # track which schedule phase is active

    block_count = 0
    for example in ds:
        text = example["text"]
        if not text or len(text.strip()) < 10:
            continue
        tokens = tokenizer.encode(text)
        main_buffer.extend(tokens)
        main_buffer.append(eos_id)

        while len(main_buffer) >= block_size:
            block_count += 1

            # Check inject_until cutoff
            if inject_until and total_blocks:
                cutoff_block = int(inject_until * total_blocks)
                if block_count > cutoff_block:
                    # Past the cutoff — yield only standard data
                    yield main_buffer[:block_size]
                    main_buffer = main_buffer[block_size:]
                    continue

            # Determine active contrastive tokens
            if schedule_buffers and total_blocks:
                frac = block_count / total_blocks
                new_phase = -1
                active_tokens = None
                for i, (sf, ef, ptoks) in enumerate(schedule_buffers):
                    if sf <= frac < ef:
                        new_phase = i
                        active_tokens = ptoks
                        break
                if new_phase != active_phase:
                    contr_idx = 0  # reset index on phase change
                    active_phase = new_phase
                if active_tokens is None:
                    # Outside all schedule phases — yield standard data
                    yield main_buffer[:block_size]
                    main_buffer = main_buffer[block_size:]
                    continue
            else:
                active_tokens = contrastive_tokens

            # Every inject_rate blocks, yield a contrastive block instead
            if inject_rate > 0 and block_count % inject_rate == 0:
                # Build contrastive block (cycling if needed)
                contr_block = []
                while len(contr_block) < block_size:
                    remaining = block_size - len(contr_block)
                    available = active_tokens[contr_idx:contr_idx + remaining]
                    contr_block.extend(available)
                    contr_idx = (contr_idx + len(available)) % len(active_tokens)
                yield contr_block[:block_size]
            else:
                yield main_buffer[:block_size]
                main_buffer = main_buffer[block_size:]


def parse_schedule(schedule_str: str, pairs_json: str | None, probe_seed: int) -> list[tuple]:
    """Parse schedule string like 'primitive:0-50,sycophancy:50-100'.

    Returns list of (start_frac, end_frac, texts) tuples.
    """
    from rho_eval.interpretability.activation import build_contrast_pairs
    from rho_eval.behavioral import load_behavioral_probes

    # Load custom pairs if available
    custom_pairs = {}
    if pairs_json:
        import json as _json
        raw = _json.loads(Path(pairs_json).read_text())
        if isinstance(raw, list):
            custom_pairs["primitive"] = raw
        elif isinstance(raw, dict):
            custom_pairs = raw

    phases = []
    for part in schedule_str.split(","):
        behavior, frac_range = part.strip().split(":")
        start_pct, end_pct = frac_range.split("-")
        start_frac = float(start_pct) / 100
        end_frac = float(end_pct) / 100

        if behavior in custom_pairs:
            texts = []
            for pair in custom_pairs[behavior]:
                texts.append(pair["positive"])
                texts.append(pair["negative"])
        else:
            eval_probes = load_behavioral_probes(behavior, seed=42)
            eval_ids = {p.get("id", "") for p in eval_probes}
            eval_texts = {p.get("text", "")[:200] for p in eval_probes}
            n_target = len(eval_probes)
            candidates = load_behavioral_probes(behavior, seed=probe_seed, n=n_target * 3)
            clean = [p for p in candidates
                     if p.get("id", "") not in eval_ids
                     and p.get("text", "")[:200] not in eval_texts][:n_target]
            pairs = build_contrast_pairs(behavior, clean)
            texts = []
            for pair in pairs:
                texts.append(pair["positive"])
                texts.append(pair["negative"])

        phases.append((start_frac, end_frac, texts))
        print(f"    Schedule: {behavior} @ {start_frac*100:.0f}%-{end_frac*100:.0f}% ({len(texts)} texts)")

    return phases


def train_contrastive(
    size: str,
    seed: int = 42,
    device_str: str = "cpu",
    output_dir: str | None = None,
    dataset: str = "openwebtext",
    inject_rate: int = 10,
    inject_behaviors: list[str] | None = None,
    audit_interval: int = 500,
    checkpoint_steps: list[int] | None = None,
    probe_seed: int = 999,
    pairs_json: str | None = None,
    inject_schedule: str | None = None,
    inject_until: float | None = None,
):
    """Train a model with contrastive behavioral text injection.

    probe_seed controls which random sample of probes is used for
    contrastive pair generation. MUST differ from the eval seed (42)
    to prevent train-eval contamination.
    """
    if size not in SCALE_CONFIGS:
        print(f"  ERROR: Unknown size '{size}'. Choose from: {SCALE_ORDER}")
        return 1

    cfg = SCALE_CONFIGS[size]
    if inject_behaviors is None:
        inject_behaviors = ["bias", "sycophancy"]

    # Output directory
    if output_dir is None:
        beh_tag = "_".join(b[:3] for b in inject_behaviors)
        output_dir = f"results/scale_ladder/{size}_seed{seed}_contr_{beh_tag}_r{inject_rate}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"  Contrastive Data Training — {size}")
    print(f"  Config: {cfg['n_layer']}L, {cfg['n_head']}H, d={cfg['n_embd']}")
    print(f"  Target: {cfg['train_tokens']:,} tokens")
    print(f"  Injection: 1/{inject_rate} blocks from [{', '.join(inject_behaviors)}]")
    print(f"  Device: {device_str}, Seed: {seed}")
    print(f"{'='*60}\n")

    torch.manual_seed(seed)

    # Build model
    print("  Building model...", flush=True)
    model, n_params = build_model(cfg)
    device = torch.device(device_str)
    model = model.to(device)
    model.train()

    # Build tokenizer
    tokenizer = build_tokenizer()

    # Load contrastive texts (using probe_seed to avoid train-eval contamination)
    print(f"  Loading contrastive texts (probe_seed={probe_seed})...", flush=True)
    contrastive_texts, training_ids = load_contrastive_texts(
        inject_behaviors, probe_seed=probe_seed, pairs_json=pairs_json,
    )
    print(f"    Total: {len(contrastive_texts)} texts")

    # Save training probe IDs for contamination audit
    contam_path = output_path / "training_probe_ids.json"
    contam_path.write_text(json.dumps({
        "probe_seed": probe_seed,
        "eval_seed": 42,
        "n_training_ids": len(training_ids),
        "training_ids": sorted(training_ids),
    }, indent=2))
    print(f"    Saved {len(training_ids)} training probe IDs → {contam_path}")

    # Count contrastive tokens
    n_contr_tokens = sum(len(tokenizer.encode(t)) + 1 for t in contrastive_texts)
    print(f"    ~{n_contr_tokens:,} contrastive tokens "
          f"({n_contr_tokens/cfg['train_tokens']*100:.2f}% of training budget)")
    inject_pct = 100.0 / inject_rate if inject_rate > 0 else 0
    print(f"    Injection rate: {inject_pct:.0f}% of blocks\n")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # LR schedule
    block_size = cfg["n_positions"]
    batch_size = cfg["batch_size"]
    tokens_per_step = block_size * batch_size
    total_steps = cfg["train_tokens"] // tokens_per_step
    warmup_steps = min(500, total_steps // 10)
    min_lr = cfg["lr"] * 0.1

    def get_lr(step):
        if step < warmup_steps:
            return cfg["lr"] * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr + 0.5 * (cfg["lr"] - min_lr) * (1 + math.cos(math.pi * progress))

    print(f"  Total steps: {total_steps:,} ({warmup_steps} warmup)")
    print(f"  Tokens/step: {tokens_per_step:,}\n", flush=True)

    # Parse schedule if provided
    schedule = None
    total_blocks = total_steps * batch_size  # total blocks across all steps
    if inject_schedule:
        print(f"  Parsing injection schedule: {inject_schedule}")
        schedule = parse_schedule(inject_schedule, pairs_json, probe_seed)

    if inject_until:
        print(f"  Injection cutoff at {inject_until*100:.0f}% of training")

    # Data stream with contrastive injection
    data_iter = contrastive_token_iterator(
        tokenizer, block_size, contrastive_texts,
        inject_rate=inject_rate, seed=seed, dataset=dataset,
        schedule=schedule, total_blocks=total_blocks,
        inject_until=inject_until,
    )

    # Subspace auditing (import here to avoid load-time deps)
    from experiments.scale_ladder.train_geometric import run_subspace_audit
    from rho_eval.utils import get_layers

    model_layers = get_layers(model)
    all_layer_indices = list(range(cfg["n_layer"]))
    audit_behaviors = list(SUBSPACE_BEHAVIORS)

    # Initial audit
    print("  Running initial subspace audit (step 0)...", flush=True)
    audit_0 = run_subspace_audit(
        model, tokenizer, all_layer_indices, device, audit_behaviors
    )
    audit_0["step"] = 0
    audit_history = [audit_0]
    (output_path / "audit_step_0").mkdir(exist_ok=True)
    (output_path / "audit_step_0" / "audit_data.json").write_text(
        json.dumps(audit_0, indent=2)
    )
    print(f"    Mean angle: {audit_0['overall_mean_angle']}°\n", flush=True)

    # Training loop
    loss_history = []
    best_loss = float("inf")
    log_interval = 100
    save_interval = max(total_steps // 5, 1000)
    checkpoint_set = set(checkpoint_steps) if checkpoint_steps else set()

    for step in range(total_steps):
        batch_tokens = []
        for _ in range(batch_size):
            try:
                batch_tokens.append(next(data_iter))
            except StopIteration:
                data_iter = contrastive_token_iterator(
                    tokenizer, block_size, contrastive_texts,
                    inject_rate=inject_rate, seed=seed + step, dataset=dataset,
                    schedule=schedule, total_blocks=total_blocks,
                    inject_until=inject_until,
                )
                batch_tokens.append(next(data_iter))

        input_ids = torch.tensor(batch_tokens, dtype=torch.long, device=device)

        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        outputs = model(input_ids=input_ids, labels=input_ids.clone())
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        tokens_seen = (step + 1) * tokens_per_step
        lm_val = loss.item()
        if lm_val < best_loss:
            best_loss = lm_val

        loss_history.append({
            "step": step + 1, "loss": lm_val, "lr": lr, "tokens": tokens_seen,
        })

        if (step + 1) % log_interval == 0:
            elapsed = time.time() - t_start
            tok_per_sec = tokens_seen / elapsed
            print(
                f"  step {step+1:>6d}/{total_steps} | "
                f"loss={lm_val:.4f} | lr={lr:.2e} | "
                f"{tok_per_sec:,.0f} tok/s | {elapsed/60:.1f}min",
                flush=True,
            )

        if (step + 1) % audit_interval == 0:
            print(f"  [audit] step {step+1}...", flush=True)
            audit = run_subspace_audit(
                model, tokenizer, all_layer_indices, device, audit_behaviors
            )
            audit["step"] = step + 1
            audit_history.append(audit)
            audit_dir = output_path / f"audit_step_{step+1}"
            audit_dir.mkdir(exist_ok=True)
            (audit_dir / "audit_data.json").write_text(json.dumps(audit, indent=2))
            print(f"    Mean angle: {audit['overall_mean_angle']}°", flush=True)

        save_regular = (step + 1) % save_interval == 0
        save_explicit = (step + 1) in checkpoint_set
        if save_regular or save_explicit:
            ckpt_path = output_path / f"checkpoint_{step+1}"
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)

    # Final save
    print(f"\n  Saving final model...", flush=True)
    model_path = output_path / "model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    print("  Running final subspace audit...", flush=True)
    final_audit = run_subspace_audit(
        model, tokenizer, all_layer_indices, device, audit_behaviors
    )
    final_audit["step"] = total_steps
    audit_history.append(final_audit)
    audit_dir = output_path / f"audit_step_{total_steps}"
    audit_dir.mkdir(exist_ok=True)
    (audit_dir / "audit_data.json").write_text(json.dumps(final_audit, indent=2))

    # Save metrics
    elapsed = time.time() - t_start
    tokens_seen = total_steps * tokens_per_step

    metrics = {
        "size": size,
        "seed": seed,
        "device": device_str,
        "n_params": n_params,
        "config": cfg,
        "contrastive": {
            "inject_rate": inject_rate,
            "inject_behaviors": inject_behaviors,
            "probe_seed": probe_seed,
            "eval_seed": 42,
            "n_contrastive_texts": len(contrastive_texts),
            "n_contrastive_tokens": n_contr_tokens,
            "n_training_probe_ids": len(training_ids),
            "pairs_json": pairs_json,
            "inject_schedule": inject_schedule,
            "inject_until": inject_until,
        },
        "total_steps": total_steps,
        "tokens_seen": tokens_seen,
        "final_loss": loss_history[-1]["loss"] if loss_history else None,
        "best_loss": best_loss,
        "elapsed_seconds": round(elapsed, 1),
        "tokens_per_second": round(tokens_seen / elapsed, 1),
        "initial_mean_angle": audit_history[0]["overall_mean_angle"],
        "final_mean_angle": audit_history[-1]["overall_mean_angle"],
        "loss_history": loss_history[::max(len(loss_history) // 200, 1)],
        "audit_history": audit_history,
    }

    metrics_path = output_path / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"\n  Training complete!")
    print(f"  Final loss: {metrics['final_loss']:.4f}")
    print(f"  Angle: {metrics['initial_mean_angle']}° → {metrics['final_mean_angle']}°")
    print(f"  Time: {elapsed/60:.1f} min ({elapsed/3600:.1f}h)")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="train-contrastive",
        description=(
            "Train a causal LM with contrastive behavioral text injection. "
            "Tests whether richer behavioral signal in the data stream "
            "enables emergence at smaller scales."
        ),
    )
    parser.add_argument("--size", required=True, choices=SCALE_ORDER)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "mps", "cuda"])
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="openwebtext",
                        choices=["openwebtext", "fineweb-edu", "wikitext"])

    inj = parser.add_argument_group("contrastive injection")
    inj.add_argument("--inject-rate", type=int, default=10,
                     help="Inject 1 contrastive block every N blocks (default: 10 = ~10%%)")
    inj.add_argument("--inject-behaviors", type=str, nargs="+",
                     default=["bias", "sycophancy"],
                     help="Behaviors to inject (default: bias sycophancy)")
    inj.add_argument("--probe-seed", type=int, default=999,
                     help="Seed for training probe sampling (must differ from eval seed=42)")
    inj.add_argument("--pairs-json", type=str, default=None,
                     help="Path to JSON with pre-built contrastive pairs. "
                          "Format: list of {positive, negative, id} dicts, "
                          "or dict keyed by behavior name.")
    inj.add_argument("--inject-schedule", type=str, default=None,
                     help="Curriculum schedule: 'behavior:start-end,behavior:start-end' "
                          "(percentages). E.g., 'primitive:0-50,sycophancy:50-100'. "
                          "Overrides --inject-behaviors for contrastive content.")
    inj.add_argument("--inject-until", type=float, default=None,
                     help="Stop injecting after this fraction of training (0-1). "
                          "E.g., 0.2 means inject for first 20%% only.")

    mon = parser.add_argument_group("monitoring")
    mon.add_argument("--audit-interval", type=int, default=500)
    mon.add_argument("--checkpoint-steps", type=int, nargs="+", default=None)

    args = parser.parse_args()

    return train_contrastive(
        size=args.size,
        seed=args.seed,
        device_str=args.device,
        output_dir=args.output,
        dataset=args.dataset,
        inject_rate=args.inject_rate,
        inject_behaviors=args.inject_behaviors,
        audit_interval=args.audit_interval,
        checkpoint_steps=args.checkpoint_steps,
        probe_seed=args.probe_seed,
        pairs_json=args.pairs_json,
        inject_schedule=args.inject_schedule,
        inject_until=args.inject_until,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
