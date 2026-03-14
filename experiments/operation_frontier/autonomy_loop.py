#!/usr/bin/env python3
"""
autonomy_loop.py — Closed-loop autonomy runner for NoetherSolve.

Given a problem YAML, this script:
  1. Expands expression templates from hunt_config (all parameter combinations)
  2. Numerically checks each candidate (vortex_checker or conservation_checker)
  3. For each numerical pass: calls Claude API to generate an oracle question,
     then runs the oracle to see if the base model already knows this structure
  4. For oracle failures: calls Claude API to generate 25 training examples,
     trains a domain adapter, re-evaluates oracle
  5. Publishes all results to results/candidates.tsv

This is the "people propose their research, system takes it from there" runner.

Usage:
    # Full autonomous loop (requires ANTHROPIC_API_KEY + MLX model)
    python autonomy_loop.py --problem problems/vortex_pair_conservation.yaml

    # Dry run — numerical sweep only, no API calls, no model load
    python autonomy_loop.py --problem problems/vortex_pair_conservation.yaml --dry-run

    # Oracle only, skip adapter training
    python autonomy_loop.py --problem problems/vortex_pair_conservation.yaml --skip-training

    # Limit oracle budget
    python autonomy_loop.py --problem problems/vortex_pair_conservation.yaml --budget 20

Prerequisites:
    export ANTHROPIC_API_KEY="sk-ant-..."
    pip install anthropic
    # Apple Silicon: mlx mlx-lm already required for oracle_wrapper.py
    # Linux/CUDA:    pip install torch transformers accelerate

Hardware:
    Apple Silicon: MLX backend (oracle_wrapper / train_vortex_adapter path)
    Linux/CUDA:    set --backend torch  (uses noethersolve_torch.py)
"""

import argparse
import csv
import datetime
import itertools
import json
import os
import re
import sys
import time

import numpy as np
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Path setup (works regardless of where the repo is cloned)
# ─────────────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments", "snap_on"))
sys.path.insert(0, _HERE)

CANDIDATES_TSV = os.path.join(_HERE, "results", "candidates.tsv")
ADAPTERS_DIR   = os.path.join(_HERE, "adapters")
os.makedirs(ADAPTERS_DIR, exist_ok=True)
os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Problem loading
# ─────────────────────────────────────────────────────────────────────────────

def load_problem(path: str) -> dict:
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Expression sweep expansion
# ─────────────────────────────────────────────────────────────────────────────

def expand_expressions(hunt_config: dict) -> list:
    """Expand expression templates with all parameter combinations.

    Templates use {alpha}, {beta}, {gamma} placeholders.
    Templates with no placeholders are included as-is.
    """
    templates    = hunt_config.get("expression_templates", [])
    alpha_range  = hunt_config.get("alpha_range", [0.1, 0.5, 1.0])
    beta_range   = hunt_config.get("beta_range",  alpha_range)
    gamma_range  = hunt_config.get("gamma_range", alpha_range)

    expressions = []
    for tmpl in templates:
        needs = {
            "alpha": "{alpha}" in tmpl,
            "beta":  "{beta}"  in tmpl,
            "gamma": "{gamma}" in tmpl,
        }
        if not any(needs.values()):
            expressions.append(tmpl)
            continue

        # Build list of (name, values) for active params
        param_pairs = []
        if needs["alpha"]: param_pairs.append(("alpha", alpha_range))
        if needs["beta"]:  param_pairs.append(("beta",  beta_range))
        if needs["gamma"]: param_pairs.append(("gamma", gamma_range))

        names  = [p[0] for p in param_pairs]
        ranges = [p[1] for p in param_pairs]
        for combo in itertools.product(*ranges):
            kwargs = dict(zip(names, combo))
            expressions.append(tmpl.format(**kwargs))

    # Deduplicate, preserve order
    seen, result = set(), []
    for e in expressions:
        if e not in seen:
            seen.add(e)
            result.append(e)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Checker dispatch
# ─────────────────────────────────────────────────────────────────────────────

def detect_checker_type(problem: dict) -> str:
    name = problem.get("name", "").lower()
    desc = problem.get("description", "").lower()
    if "vortex" in name or "vortex" in desc or "kirchhoff" in desc:
        return "vortex"
    if "3body" in name or "three" in name or "gravitational" in desc or "nbody" in name:
        return "3body"
    return problem.get("checker_type", "vortex")


def frac_var_safe(arr) -> float:
    arr = np.asarray(arr, dtype=float)
    mean = np.mean(arr)
    if abs(mean) < 1e-15:
        return float(np.std(arr))
    return float(np.std(arr) / abs(mean))


def check_expression_vortex(expr: str, ic_names: list) -> dict:
    """Run vortex checker on each IC. Returns {ic_name: frac_var}."""
    from vortex_checker import IC_MAP, integrate_vortex, parse_state
    results = {}
    for ic_name in ic_names:
        if ic_name not in IC_MAP:
            continue
        try:
            G, pos0 = IC_MAP[ic_name]()
            t, state = integrate_vortex(G, pos0)
            s = parse_state(t, state, G)
            vals = eval(expr, {"s": s, "np": np})  # noqa: S307
            results[ic_name] = frac_var_safe(vals)
        except Exception as exc:  # noqa: BLE001
            results[ic_name] = float("nan")
            _ = exc
    return results


def check_expression_3body(expr: str, ic_names: list) -> dict:
    """Run 3-body checker on each IC. Returns {ic_name: frac_var}."""
    from conservation_checker import IC_MAP, integrate_3body
    results = {}
    for ic_name in ic_names:
        if ic_name not in IC_MAP:
            continue
        try:
            ic_data = IC_MAP[ic_name]()
            if len(ic_data) == 3:
                masses, pos0, vel0 = ic_data
                t, state = integrate_3body(masses, pos0, vel0)
            else:
                masses, pos0 = ic_data
                import numpy as np
                vel0 = np.zeros_like(pos0)
                t, state = integrate_3body(masses, pos0, vel0)

            # Try to import parse_state_3body, fall back to parse_state
            try:
                from conservation_checker import parse_state_3body as ps
            except ImportError:
                from conservation_checker import parse_state as ps
            s = ps(t, state, masses)
            vals = eval(expr, {"s": s, "np": np})  # noqa: S307
            results[ic_name] = frac_var_safe(vals)
        except Exception as exc:  # noqa: BLE001
            results[ic_name] = float("nan")
            _ = exc
    return results


def numerical_check(expr: str, checker_type: str,
                    ic_names: list, threshold: float) -> tuple:
    """Returns (passed: bool, fv_by_ic: dict)."""
    if checker_type == "vortex":
        fv_by_ic = check_expression_vortex(expr, ic_names)
    else:
        fv_by_ic = check_expression_3body(expr, ic_names)

    valid = {k: v for k, v in fv_by_ic.items() if not np.isnan(v)}
    if not valid:
        return False, fv_by_ic

    passed = all(v < threshold for v in valid.values())
    return passed, fv_by_ic


# ─────────────────────────────────────────────────────────────────────────────
# Human-readable label
# ─────────────────────────────────────────────────────────────────────────────

def expr_to_label(expr: str) -> str:
    """Convert Python expression to human-readable label."""
    label = expr.replace("s['", "").replace("']", "")
    label = label.replace("**", "^")
    label = re.sub(r"\s*\*\s*", "·", label)
    label = label.replace("np.", "")
    return label[:72] if len(label) > 72 else label


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate detection
# ─────────────────────────────────────────────────────────────────────────────

def load_tested_hypotheses() -> set:
    if not os.path.exists(CANDIDATES_TSV):
        return set()
    tested = set()
    with open(CANDIDATES_TSV) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tested.add(row.get("hypothesis", ""))
    return tested


def already_tested(expr: str, tested: set) -> bool:
    label = expr_to_label(expr)
    canonical = expr.replace(" ", "").replace("**", "^")
    for h in tested:
        h_canon = h.replace(" ", "").replace("**", "^")
        if label in h or canonical in h_canon:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Claude API: generate oracle question + training data
# ─────────────────────────────────────────────────────────────────────────────

_GENERATION_PROMPT = """\
You are generating evaluation data for NoetherSolve — a physics oracle benchmark.

SITUATION: We numerically verified that the following expression is conserved,
but a language model does not recognize it. We need:
  (A) An oracle question testing whether the model knows this is conserved.
  (B) 25 training examples to teach a small logit adapter about this domain.

EXPRESSION:   {expr}
DOMAIN:       {domain}
CHECKER:      frac_var = {fv_str}  (threshold {threshold:.1e} → NUMERICALLY CONSERVED)
IC TYPE:      {ic_name}

RETURN a single JSON object with exactly two keys:

  "oracle_question": one fact with keys "context", "truth", "distractors"
  "training_examples": list of 25 facts, each with "context", "truth", "distractors"

FORMAT RULES (violations cause oracle failure — follow exactly):
  - Compact symbolic notation ONLY. No verbose prose descriptions.
  - context: One short sentence. Example: "2D vortex, Γ₁=Γ₂=1, Γ₃=0.01. Which quantity is conserved?"
  - truth: Compact formula. Example: "Q = r₁₂ + 0.01(r₁₃+r₂₃) = const"
  - distractors: exactly 3 plausible WRONG answers using similar notation
    (wrong signs, swapped indices, missing terms, wrong exponents)

TRAINING EXAMPLES breakdown:
  - 8 examples: directly about THIS expression (vary wording, same formula)
  - 10 examples: other domain facts (H, Lz, Px, Py, standard invariants)
  - 7 examples: NEGATIVE examples (expressions that look similar but do NOT conserve)

Return ONLY the raw JSON. No markdown code fences, no explanation.
"""


def call_claude_generate(expr: str, domain_desc: str, ic_name: str,
                          fv_by_ic: dict, threshold: float,
                          anthropic_client) -> dict | None:
    """Call Claude API to generate oracle question + training examples."""
    fv_str = ", ".join(
        f"{ic}:{fv:.2e}" for ic, fv in fv_by_ic.items() if not np.isnan(fv)
    )
    prompt = _GENERATION_PROMPT.format(
        expr=expr,
        domain=domain_desc[:300],
        fv_str=fv_str,
        threshold=threshold,
        ic_name=ic_name,
    )

    try:
        import anthropic as _anth
        response = anthropic_client.messages.create(
            model="claude-opus-4-6",
            max_tokens=8192,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract text block (skip thinking blocks)
        text = ""
        for block in response.content:
            if block.type == "text":
                text = block.text
                break

        # Strip markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text.rstrip())

        data = json.loads(text)
        # Validate structure
        if "oracle_question" not in data or "training_examples" not in data:
            raise ValueError("Missing required keys in generated JSON")
        return data

    except Exception as exc:
        print(f"    [claude_api] Generation failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Oracle
# ─────────────────────────────────────────────────────────────────────────────

def load_model_mlx(model_name: str):
    """Load model with MLX backend. Returns (model, tokenizer, lm_head)."""
    import mlx_lm
    import experiments.operation_destroyer.train_v3 as t3
    t0 = time.time()
    print(f"  Loading {model_name} (MLX)...")
    model, tokenizer = mlx_lm.load(model_name)
    model.freeze()
    lm_head = t3.get_lm_head_fn(model)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return model, tokenizer, lm_head


def run_oracle_on_facts(model, tokenizer, facts: list,
                         adapter=None, lm_head=None) -> dict:
    """Score oracle facts. Returns {n_pass, n_total, mean_margin, margins}."""
    from experiments.operation_destroyer.eval_mc import score_fact_mc
    margins, wins = [], 0
    for fact in facts:
        win, margin, _, _ = score_fact_mc(
            model, tokenizer,
            fact["context"], fact["truth"], fact["distractors"],
            adapter=adapter, lm_head=lm_head,
        )
        margins.append(float(margin))
        wins += int(win)
    return {
        "n_pass":      wins,
        "n_total":     len(facts),
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
        "margins":     margins,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Adapter training
# ─────────────────────────────────────────────────────────────────────────────

def train_on_examples(model, tokenizer, lm_head,
                       examples: list, steps: int, lr: float,
                       d_inner: int, margin_target: float = 1.5):
    """Train logit adapter on generated examples list.

    examples: list of dicts with keys "context", "truth", "distractors"
    Returns trained adapter object (MLX).
    """
    from train_vortex_adapter import train_vortex_adapter

    # Convert dict format → (ctx, truth, distractors, domain) tuples
    tuples = []
    for ex in examples:
        ctx         = ex.get("context", "")
        truth       = ex.get("truth", ex.get("completion", ""))
        distractors = ex.get("distractors", [])
        # Ensure we have at least 3 distractors (pad with generic if needed)
        while len(distractors) < 3:
            distractors.append("This quantity is not conserved")
        tuples.append((ctx, truth, distractors[:4], "auto"))

    return train_vortex_adapter(
        model, tokenizer, lm_head, tuples,
        steps=steps, lr=lr, d_inner=d_inner, margin_target=margin_target,
    )


def save_adapter_mlx(adapter, path: str):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    weights = dict(tree_flatten(adapter.parameters()))
    mx.savez(path, **weights)


def load_adapter_mlx(path: str, model, d_inner: int):
    """Load saved adapter weights into a new adapter object."""
    import mlx.core as mx
    from experiments.snap_on.module import SnapOnConfig, create_adapter

    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model    = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    cfg     = SnapOnConfig(d_model=d_model, d_inner=d_inner, n_layers=0,
                           n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(cfg)
    weights = mx.load(path)
    adapter.load_weights(list(weights.items()))
    mx.eval(adapter.parameters())
    return adapter


# ─────────────────────────────────────────────────────────────────────────────
# TSV publishing
# ─────────────────────────────────────────────────────────────────────────────

def append_tsv_row(expr: str, ic_name: str, fv_by_ic: dict,
                   margin_baseline, margin_adapted,
                   verdict: str, classification: str,
                   n_pass_str: str = "1/1"):
    today      = datetime.date.today().isoformat()
    hypothesis = f"{expr_to_label(expr)} on {ic_name} (auto)"

    if margin_baseline is None:
        margin_str = "pending"
    elif margin_adapted is not None:
        margin_str = f"{margin_baseline:+.4f}→{margin_adapted:+.4f}"
    else:
        margin_str = f"{margin_baseline:+.4f}"

    fv_str = ";".join(
        f"{ic}:{fv:.2e}" for ic, fv in fv_by_ic.items() if not np.isnan(fv)
    )
    full_class = f"{classification} {fv_str}".strip()

    row = [today, hypothesis, margin_str, n_pass_str, verdict, full_class]
    with open(CANDIDATES_TSV, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(row)
    print(f"    → TSV: {verdict}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NoetherSolve autonomy loop — sweep → oracle → adapter → publish"
    )
    parser.add_argument("--problem",        required=True,
                        help="Path to problem .yaml or .json")
    parser.add_argument("--budget",         type=int,   default=200,
                        help="Max oracle evaluations before stopping (default: 200)")
    parser.add_argument("--dry-run",        action="store_true",
                        help="Numerical sweep only — no model load, no API calls")
    parser.add_argument("--skip-oracle",    action="store_true",
                        help="Skip oracle + training (record checker results only)")
    parser.add_argument("--skip-training",  action="store_true",
                        help="Run oracle but skip adapter training step")
    parser.add_argument("--adapter-steps",  type=int,   default=1000)
    parser.add_argument("--adapter-lr",     type=float, default=1e-5)
    parser.add_argument("--d-inner",        type=int,   default=64)
    parser.add_argument("--margin-target",  type=float, default=1.5)
    parser.add_argument("--model",          default=None,
                        help="Override model from problem YAML")
    parser.add_argument("--backend",        choices=["mlx", "torch"], default="mlx")
    parser.add_argument("--no-publish",     action="store_true",
                        help="Don't write results to candidates.tsv")
    args = parser.parse_args()

    # ── Load problem ──────────────────────────────────────────────────────────
    problem_path = os.path.abspath(args.problem)
    problem_dir  = os.path.dirname(problem_path)
    problem      = load_problem(problem_path)

    hunt_cfg     = problem.get("hunt_config", {})
    model_name   = args.model or problem.get("model", "Qwen/Qwen3-4B-Base")
    threshold    = float(hunt_cfg.get("numerical_threshold", 0.005))
    ic_priority  = hunt_cfg.get("ic_priority", ["equal_pair"])
    checker_type = detect_checker_type(problem)
    domain_desc  = (problem.get("description") or problem.get("name") or
                    "Unknown physics domain")

    print(f"\n{'='*72}")
    print(f"  NoetherSolve Autonomy Loop")
    print(f"{'='*72}")
    print(f"  Problem:     {problem.get('name', args.problem)}")
    print(f"  Checker:     {checker_type}")
    print(f"  ICs:         {ic_priority}")
    print(f"  Threshold:   frac_var < {threshold:.1e}")
    print(f"  Oracle:      {model_name}")
    print(f"  Budget:      {args.budget} oracle calls")
    print(f"  Dry run:     {args.dry_run}")
    print(f"  Skip oracle: {args.skip_oracle}")
    print(f"  Skip train:  {args.skip_training}")
    print(f"  Publish:     {not args.no_publish}")

    # ── Expand expression templates ───────────────────────────────────────────
    all_exprs = expand_expressions(hunt_cfg)
    tested    = load_tested_hypotheses()
    print(f"\n  Expressions to test: {len(all_exprs)}")
    print(f"  Already in TSV:      {len(tested)}")

    # ── Phase 1: Numerical sweep ──────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  PHASE 1 — Numerical sweep  (frac_var threshold: {threshold:.1e})")
    print(f"{'─'*72}")

    numerical_passes = []   # [(expr, ic_name, fv_by_ic)]
    skipped = 0

    for expr in all_exprs:
        if already_tested(expr, tested):
            skipped += 1
            continue

        passed, fv_by_ic = numerical_check(expr, checker_type, ic_priority, threshold)
        label   = expr_to_label(expr)
        fv_str  = "  ".join(
            f"{ic}:{fv:.2e}" for ic, fv in fv_by_ic.items() if not np.isnan(fv)
        )
        verdict = "PASS ✓" if passed else "fail"
        print(f"  {verdict}  {label:56s}  {fv_str}")

        if passed:
            numerical_passes.append((expr, ic_priority[0], fv_by_ic))
        elif not args.dry_run and not args.skip_oracle and not args.no_publish:
            # Record CHECKER-FAIL
            fv_note = ";".join(
                f"{ic}:{fv:.2e}" for ic, fv in fv_by_ic.items() if not np.isnan(fv)
            )
            append_tsv_row(expr, ic_priority[0], fv_by_ic,
                           None, None, "CHECKER-FAIL", fv_note)

    print(f"\n  Numerical passes:  {len(numerical_passes)}")
    print(f"  Skipped (dup):     {skipped}")
    print(f"  Fails:             {len(all_exprs) - skipped - len(numerical_passes)}")

    if args.dry_run or args.skip_oracle:
        print(f"\n  [dry-run / skip-oracle] Stopping after numerical sweep.")
        if numerical_passes:
            print(f"\n  Passes ready for oracle:")
            for expr, ic, fv_by_ic in numerical_passes:
                print(f"    {expr_to_label(expr)}")
        return

    if not numerical_passes:
        print("\n  No numerical passes found. Nothing to oracle-test. Done.")
        return

    # ── Phase 2: Oracle + training loop ──────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  PHASE 2 — Oracle evaluation + adapter training")
    print(f"{'─'*72}")

    # Load model
    if args.backend == "mlx":
        model, tokenizer, lm_head = load_model_mlx(model_name)
    else:
        # PyTorch / CUDA backend
        print(f"  Loading {model_name} (PyTorch)...")
        try:
            from noethersolve_torch import load_model_cpu_or_cuda
            model, tokenizer, lm_head = load_model_cpu_or_cuda(model_name)
        except ImportError:
            print("  ERROR: noethersolve_torch.py not found or import failed.")
            print("  On Apple Silicon use --backend mlx (default).")
            return

    # Load Anthropic client
    anthropic_client = None
    if not args.skip_training:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("\n  ⚠  ANTHROPIC_API_KEY not set.")
            print("     Training data generation disabled.")
            print("     Set with: export ANTHROPIC_API_KEY='sk-ant-...'")
            print("     Or use --skip-training to oracle without adapter repair.\n")
            args.skip_training = True
        else:
            try:
                import anthropic as _anth
                anthropic_client = _anth.Anthropic(api_key=api_key)
                print(f"  Anthropic client ready (claude-opus-4-6).")
            except ImportError:
                print("  ⚠  anthropic not installed. pip install anthropic")
                args.skip_training = True

    # ── Per-candidate loop ────────────────────────────────────────────────────
    oracle_count = 0
    dual_pass    = []    # (expr, margin, fv_by_ic)
    flipped      = []    # (expr, b_margin, r_margin, fv_by_ic, adapter_path)
    open_gaps    = []    # (expr, margin, fv_by_ic)

    for expr, ic_name, fv_by_ic in numerical_passes:
        if oracle_count >= args.budget:
            print(f"\n  Budget reached ({args.budget} oracle calls). Stopping.")
            break

        label  = expr_to_label(expr)
        fv_min = min(v for v in fv_by_ic.values() if not np.isnan(v))
        print(f"\n  ┌─ Candidate: {label}")
        print(f"  │  IC: {ic_name}  frac_var: {fv_min:.2e}")

        # ── Step A: Generate oracle question via Claude API ──────────────────
        oracle_facts      = None
        training_examples = []

        if anthropic_client:
            print(f"  │  Calling claude-opus-4-6 to generate oracle question + training data...")
            generated = call_claude_generate(
                expr, domain_desc, ic_name, fv_by_ic, threshold, anthropic_client
            )
            if generated:
                oracle_facts      = [generated["oracle_question"]]
                training_examples = generated.get("training_examples", [])
                print(f"  │  Generated: 1 oracle question, {len(training_examples)} training examples")

        # Fallback: use domain verification set
        if oracle_facts is None:
            vs_relpath = problem.get("verification_set", "")
            vs_path    = (vs_relpath if os.path.isabs(vs_relpath)
                          else os.path.join(problem_dir, vs_relpath))
            if vs_path and os.path.exists(vs_path):
                with open(vs_path) as f:
                    vdata = json.load(f)
                facts_list = vdata if isinstance(vdata, list) else vdata.get("facts", [])
                oracle_facts = facts_list
                print(f"  │  Fallback: domain verification set ({len(oracle_facts)} facts)")
            else:
                print(f"  │  No oracle facts available — skipping this candidate")
                continue

        # ── Step B: Baseline oracle ──────────────────────────────────────────
        baseline = run_oracle_on_facts(model, tokenizer, oracle_facts,
                                       adapter=None, lm_head=lm_head)
        oracle_count += 1
        b_margin  = baseline["mean_margin"]
        pass_rate = f"{baseline['n_pass']}/{baseline['n_total']}"
        print(f"  │  Baseline oracle: margin={b_margin:+.3f}  ({pass_rate} pass)")

        if b_margin > 0:
            print(f"  └─ ✓ DUAL-PASS — model already knows this structure!")
            dual_pass.append((expr, b_margin, fv_by_ic))
            if not args.no_publish:
                append_tsv_row(expr, ic_name, fv_by_ic, b_margin, None,
                               "DUAL-PASS",
                               f"frac_var={fv_min:.2e}",
                               n_pass_str=pass_rate)
            continue

        # Oracle failed
        print(f"  │  Oracle FAIL (margin={b_margin:+.3f})")

        if args.skip_training or not anthropic_client or not training_examples:
            verdict = "ORACLE-FAIL+CHECKER-PASS"
            note    = f"margin={b_margin:.2f} frac_var={fv_min:.2e}"
            open_gaps.append((expr, b_margin, fv_by_ic))
            if not args.no_publish:
                append_tsv_row(expr, ic_name, fv_by_ic, b_margin, None,
                               verdict, note, n_pass_str=pass_rate)
            print(f"  └─ Recorded as {verdict}")
            continue

        # ── Step C: Train adapter ──────────────────────────────────────────
        print(f"  │  Training adapter ({len(training_examples)} examples, "
              f"{args.adapter_steps} steps)...")
        adapter_name = (f"auto_{ic_name}_"
                        f"{datetime.datetime.now().strftime('%m%d_%H%M%S')}.npz")
        adapter_path = os.path.join(ADAPTERS_DIR, adapter_name)

        try:
            adapter = train_on_examples(
                model, tokenizer, lm_head,
                training_examples,
                steps=args.adapter_steps,
                lr=args.adapter_lr,
                d_inner=args.d_inner,
                margin_target=args.margin_target,
            )
            save_adapter_mlx(adapter, adapter_path)
            print(f"  │  Adapter saved: {adapter_name}")
        except Exception as exc:
            print(f"  │  Training error: {exc}")
            open_gaps.append((expr, b_margin, fv_by_ic))
            if not args.no_publish:
                append_tsv_row(expr, ic_name, fv_by_ic, b_margin, None,
                               "ORACLE-FAIL+CHECKER-PASS",
                               f"training_failed frac_var={fv_min:.2e}",
                               n_pass_str=pass_rate)
            print(f"  └─ Training failed — recorded as open gap")
            continue

        # ── Step D: Re-evaluate with adapter ──────────────────────────────
        print(f"  │  Re-evaluating oracle with adapter...")
        repaired = run_oracle_on_facts(model, tokenizer, oracle_facts,
                                       adapter=adapter, lm_head=lm_head)
        oracle_count += 1
        r_margin  = repaired["mean_margin"]
        delta     = r_margin - b_margin
        pass_rate2 = f"{repaired['n_pass']}/{repaired['n_total']}"
        print(f"  │  Repaired oracle: margin={r_margin:+.3f}  "
              f"(Δ={delta:+.3f})  ({pass_rate2} pass)")

        # Diagnose quadrant
        from oracle_wrapper import diagnose_quadrant
        quadrant = diagnose_quadrant(b_margin, r_margin, checker_passed=True)

        if r_margin > 0:
            print(f"  └─ 🎉 QUADRANT3→FLIPPED!  {b_margin:+.3f} → {r_margin:+.3f}")
            flipped.append((expr, b_margin, r_margin, fv_by_ic, adapter_path))
            if not args.no_publish:
                append_tsv_row(expr, ic_name, fv_by_ic, b_margin, r_margin,
                               "QUADRANT3→FLIPPED",
                               f"auto_adapter_{adapter_name} frac_var={fv_min:.2e}",
                               n_pass_str=pass_rate2)
        else:
            quad_upper = quadrant.upper()
            print(f"  └─ Quadrant: {quad_upper}  (margin still {r_margin:+.3f})")
            open_gaps.append((expr, b_margin, fv_by_ic))
            if not args.no_publish:
                append_tsv_row(expr, ic_name, fv_by_ic, b_margin, r_margin,
                               "ORACLE-FAIL+CHECKER-PASS",
                               f"{quad_upper} margin={b_margin:.2f}→{r_margin:.2f}",
                               n_pass_str=pass_rate2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  NoetherSolve Autonomy Loop — Complete")
    print(f"{'='*72}")
    print(f"  Numerical passes:    {len(numerical_passes)}")
    print(f"  Oracle calls used:   {oracle_count} / {args.budget}")
    print(f"  DUAL-PASS:           {len(dual_pass)}")
    print(f"  FLIPPED:             {len(flipped)}")
    print(f"  Open gaps:           {len(open_gaps)}")

    if dual_pass:
        print(f"\n  ✓ DUAL-PASS (model already knew):")
        for expr, margin, _ in dual_pass:
            print(f"    margin={margin:+.3f}  {expr_to_label(expr)}")

    if flipped:
        print(f"\n  🎉 FLIPPED (adapter rescued):")
        for expr, b_m, r_m, _, adp in flipped:
            print(f"    {b_m:+.3f}→{r_m:+.3f}  {expr_to_label(expr)}")
            print(f"    adapter: {os.path.basename(adp)}")

    if open_gaps:
        print(f"\n  ⚠  Open gaps (still failing after repair):")
        for expr, margin, fv_by_ic in open_gaps:
            fv_str = ";".join(
                f"{ic}:{fv:.2e}" for ic, fv in fv_by_ic.items() if not np.isnan(fv)
            )
            print(f"    margin={margin:+.3f}  {expr_to_label(expr)}")
            print(f"    {fv_str}")

    if not args.no_publish and (dual_pass or flipped or open_gaps):
        print(f"\n  Results written to: {CANDIDATES_TSV}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Open questions queue
# ─────────────────────────────────────────────────────────────────────────────

OPEN_QUESTIONS_FILE = os.path.join(_HERE, "results", "open_questions.jsonl")


def load_open_questions() -> list:
    """Load all entries from the open questions queue."""
    if not os.path.exists(OPEN_QUESTIONS_FILE):
        return []
    questions = []
    with open(OPEN_QUESTIONS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    questions.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return questions


def save_open_questions(new_questions: list, source: str = "autonomy_loop"):
    """Append new questions to the open questions queue (deduplicates by expr)."""
    existing = load_open_questions()
    existing_exprs = {q.get("expr", "") for q in existing}
    existing_descs = {q.get("description", "") for q in existing}

    added = 0
    with open(OPEN_QUESTIONS_FILE, "a") as f:
        for q in new_questions:
            expr = q.get("expr", "")
            desc = q.get("description", "")
            # Skip duplicates
            if expr and expr in existing_exprs:
                continue
            if desc and desc in existing_descs:
                continue

            q["generated_by"] = source
            q["status"]       = "open"
            q["timestamp"]    = datetime.date.today().isoformat()
            if "id" not in q:
                import hashlib
                key   = (expr or desc)[:64]
                q["id"] = "oq-" + hashlib.md5(key.encode()).hexdigest()[:8]

            f.write(json.dumps(q) + "\n")
            existing_exprs.add(expr)
            existing_descs.add(desc)
            added += 1

    return added


_PROBLEM_GENERATION_PROMPT = """\
You are a physics research assistant for NoetherSolve — a system that finds
numerically conserved quantities in dynamical systems that LLMs don't recognize.

DOMAIN: {domain}
CHECKER: {checker_type}
AVAILABLE ICs: {ic_list}

WHAT WE JUST FOUND (in this run):
  Numerical passes (conserved but oracle might not know):
{pass_list}

  Open gaps (conserved, oracle failed even after adapter repair):
{gap_list}

  DUAL-PASS (conserved, oracle already knew):
{dual_list}

YOUR TASK: Generate new research hypotheses for the NEXT sweep.
Focus on:
  1. Extensions of successful patterns (e.g., if r12+ε(r13+r23) passed, try r12²+ε·r13²)
  2. Products/ratios of conserved quantities (H·Lz, H/Lz², r12·H, etc.)
  3. Higher-order combinations and geometric invariants
  4. Physical intuition: adiabatic invariants, action variables, KAM tori
  5. Analogies to known results in related systems (figure-8 choreography, restricted 3-body)

Also generate 2-3 broader RESEARCH DIRECTIONS that would require a new problem YAML.

Return JSON with exactly two keys:
  "new_expressions": list of objects, each with:
    - "expr": Python expression using s['r12'], s['H'], s['Lz'], etc.
    - "ic": IC name from available list above
    - "rationale": one sentence why this might conserve (compact, no prose)
    - "priority": "high" | "medium" | "low"

  "research_directions": list of objects, each with:
    - "description": one sentence research question
    - "domain": same domain or "new_domain"
    - "priority": "high" | "medium" | "low"

Return ONLY raw JSON. No markdown, no explanation outside the JSON.
Maximum 12 new_expressions, maximum 3 research_directions.
"""


def generate_new_problems(
    domain_desc: str,
    checker_type: str,
    ic_priority: list,
    dual_pass: list,
    flipped: list,
    open_gaps: list,
    anthropic_client,
) -> tuple:
    """Call Claude to generate new expression hypotheses + research directions.

    Returns (new_expressions: list, research_directions: list).
    """
    def _fmt(items, kind="expr"):
        if not items:
            return "    (none)"
        lines = []
        for item in items[:8]:
            if kind == "expr":
                expr, margin, fv = item[0], item[1], item[2]
                fv_min = min(v for v in fv.values() if not np.isnan(v))
                lines.append(f"    {expr_to_label(expr)}  frac_var={fv_min:.2e}")
            else:
                lines.append(f"    {item}")
        return "\n".join(lines)

    prompt = _PROBLEM_GENERATION_PROMPT.format(
        domain=domain_desc[:400],
        checker_type=checker_type,
        ic_list=", ".join(ic_priority),
        pass_list=_fmt(dual_pass + [x[:3] for x in flipped]),
        gap_list=_fmt(open_gaps),
        dual_list=_fmt(dual_pass),
    )

    try:
        response = anthropic_client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        )
        text = ""
        for block in response.content:
            if block.type == "text":
                text = block.text
                break
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text.rstrip())
        data = json.loads(text)
        return (
            data.get("new_expressions", []),
            data.get("research_directions", []),
        )
    except Exception as exc:
        print(f"    [problem_generation] Error: {exc}")
        return [], []


def print_open_questions_menu():
    """Print the open questions queue in a human-readable format."""
    questions = load_open_questions()
    if not questions:
        print("  (no open questions in queue)")
        return

    expr_q  = [q for q in questions if q.get("type") == "expression" and q["status"] == "open"]
    dir_q   = [q for q in questions if q.get("type") == "direction"  and q["status"] == "open"]

    if expr_q:
        print(f"\n  Expression hypotheses to test ({len(expr_q)}):")
        for q in expr_q[:10]:
            priority = q.get("priority", "?")
            print(f"    [{priority}] {q.get('expr', '')}  (IC: {q.get('ic', '?')})")
            if q.get("rationale"):
                print(f"         rationale: {q['rationale']}")

    if dir_q:
        print(f"\n  Research directions ({len(dir_q)}):")
        for q in dir_q[:5]:
            priority = q.get("priority", "?")
            print(f"    [{priority}] {q.get('description', '')}")


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: show open questions
# ─────────────────────────────────────────────────────────────────────────────

def cmd_show_queue(args):
    """Print the open questions queue."""
    questions = load_open_questions()
    total     = len(questions)
    open_qs   = [q for q in questions if q.get("status") == "open"]
    print(f"\n  Open questions queue: {len(open_qs)} open / {total} total")
    print(f"  File: {OPEN_QUESTIONS_FILE}")
    print_open_questions_menu()


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: propose a new problem interactively
# ─────────────────────────────────────────────────────────────────────────────

_PROBLEM_YAML_PROMPT = """\
You are helping a researcher set up a new NoetherSolve problem.

The researcher wants to investigate the following unsolved problem:
"{user_problem}"

Your task: Generate a complete problem YAML for the NoetherSolve hunt_config.

The YAML should include:
  name: short_snake_case_name
  description: |
    2-3 sentences describing the physical system and target.
  model: "Qwen/Qwen3-4B-Base"
  oracle: "stem_margin"
  verification_set: "<name>_facts.json"
  pass_threshold: 0.8
  hunt_config:
    mode: "knowledge_gap"
    numerical_threshold: 0.005
    ic_priority:
      - <list the most relevant IC names>
    expression_templates:
      - <10-15 Python expression templates using s['...'] dict>
      - use {alpha}, {beta}, {gamma} placeholders for parameters
    alpha_range: [0.001, 0.01, 0.1, 0.5, 1.0]
  notes: |
    Physical reasoning for the expression templates.

Available variable names in s[]:
  Vortex domain: r12, r13, r23, H, Lz, Px, Py, Xcm, Ycm
  3-body domain: r12, r13, r23, H, Lz, Px, Py

Available IC names:
  Vortex: equal_pair, unequal_pair, opposite_pair, three_symmetric, three_random, restricted
  3-body: figure8, lagrange, random, hierarchical

Choose the domain (vortex or 3body) that best fits the researcher's question.

Return ONLY the raw YAML text. No markdown fences, no explanation.
"""


def cmd_propose_problem(args):
    """Interactively help user formulate an unsolved problem as a YAML."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("\n  ERROR: ANTHROPIC_API_KEY not set.")
        print("  Set with: export ANTHROPIC_API_KEY='sk-ant-...'")
        return

    import anthropic as _anth
    client = _anth.Anthropic(api_key=api_key)

    print("\n  NoetherSolve — New Problem Proposer")
    print("  ─────────────────────────────────────────────────────────────────")
    print("  Describe your unsolved physics problem. Examples:")
    print("    - 'Does the 4-vortex Z4-symmetric IC have a near-invariant?'")
    print("    - 'Is there an approximate action variable for the restricted 3-body?'")
    print("    - 'What is conserved in a near-integrable 3-vortex with Γ=[1,1,0.1]?'")
    print()

    if args.problem_text:
        user_problem = args.problem_text
        print(f"  Problem: {user_problem}")
    else:
        try:
            user_problem = input("  Your unsolved problem: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return

    if not user_problem:
        print("  No problem entered. Exiting.")
        return

    print(f"\n  Generating problem YAML for: {user_problem!r}")
    print("  Calling claude-opus-4-6 (adaptive thinking)...")

    try:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content":
                        _PROBLEM_YAML_PROMPT.format(user_problem=user_problem)}],
        )
        yaml_text = ""
        for block in response.content:
            if block.type == "text":
                yaml_text = block.text
                break

        # Strip fences if present
        yaml_text = yaml_text.strip()
        if yaml_text.startswith("```"):
            yaml_text = re.sub(r"^```[a-z]*\n?", "", yaml_text)
            yaml_text = re.sub(r"\n?```$", "", yaml_text.rstrip())

        # Parse to get the name
        parsed  = yaml.safe_load(yaml_text)
        name    = parsed.get("name", "custom_problem")
        outpath = os.path.join(_HERE, "problems", f"{name}.yaml")

        # Write YAML
        os.makedirs(os.path.join(_HERE, "problems"), exist_ok=True)
        with open(outpath, "w") as f:
            f.write(f"# Generated by autonomy_loop.py propose-problem\n")
            f.write(f"# User question: {user_problem}\n\n")
            f.write(yaml_text)

        print(f"\n  Problem YAML written: {outpath}")
        print(f"\n  Preview:\n")
        for line in yaml_text.splitlines()[:30]:
            print(f"    {line}")
        if len(yaml_text.splitlines()) > 30:
            print(f"    ... ({len(yaml_text.splitlines())} lines total)")

        print(f"\n  Next steps:")
        print(f"    1. Review and edit: {outpath}")
        print(f"    2. Create verification facts: problems/{name}_facts.json")
        print(f"       (run oracle_wrapper.py to check baseline first)")
        print(f"    3. Run autonomy loop:")
        print(f"       python autonomy_loop.py --problem problems/{name}.yaml")

        # Also add to open questions queue
        entry = {
            "type":        "direction",
            "domain":      parsed.get("name", "custom"),
            "description": user_problem,
            "yaml_file":   outpath,
            "priority":    "high",
        }
        save_open_questions([entry], source="propose-problem")
        print(f"\n  Added to open questions queue: {OPEN_QUESTIONS_FILE}")

    except Exception as exc:
        print(f"\n  Error generating YAML: {exc}")
        import traceback
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point (supports subcommands)
# ─────────────────────────────────────────────────────────────────────────────

def main():
    top = argparse.ArgumentParser(
        description="NoetherSolve autonomy loop — sweep → oracle → adapter → publish",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  run             Full autonomy loop (default when --problem is given)
  show-queue      Print the open questions queue
  propose-problem Interactively generate a problem YAML from a natural language question

Examples:
  python autonomy_loop.py --problem problems/vortex_pair_conservation.yaml
  python autonomy_loop.py show-queue
  python autonomy_loop.py propose-problem
  python autonomy_loop.py propose-problem --text "Is there a near-invariant for 4-vortex Z4 IC?"
        """,
    )
    sub = top.add_subparsers(dest="subcommand")

    # ── show-queue ────────────────────────────────────────────────────────────
    sub.add_parser("show-queue", help="Print open questions queue")

    # ── propose-problem ───────────────────────────────────────────────────────
    pp = sub.add_parser("propose-problem", help="Generate a problem YAML from a question")
    pp.add_argument("--text", dest="problem_text", default=None,
                    help="Problem description (if not given, prompts interactively)")

    # ── run (default) ─────────────────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Full autonomy loop")
    _add_run_args(run_p)

    # Support bare --problem flag (no subcommand) for backward compatibility
    top.add_argument("--problem",        default=None)
    top.add_argument("--budget",         type=int,   default=200)
    top.add_argument("--dry-run",        action="store_true")
    top.add_argument("--skip-oracle",    action="store_true")
    top.add_argument("--skip-training",  action="store_true")
    top.add_argument("--adapter-steps",  type=int,   default=1000)
    top.add_argument("--adapter-lr",     type=float, default=1e-5)
    top.add_argument("--d-inner",        type=int,   default=64)
    top.add_argument("--margin-target",  type=float, default=1.5)
    top.add_argument("--model",          default=None)
    top.add_argument("--backend",        choices=["mlx", "torch"], default="mlx")
    top.add_argument("--no-publish",     action="store_true")
    top.add_argument("--generate-problems", action="store_true",
                     help="After loop, call Claude to propose new hypotheses")

    args = top.parse_args()

    if args.subcommand == "show-queue":
        cmd_show_queue(args)
        return
    if args.subcommand == "propose-problem":
        cmd_propose_problem(args)
        return

    if not args.problem:
        top.print_help()
        return

    _run_loop(args)


def _add_run_args(p):
    p.add_argument("--problem",        required=True)
    p.add_argument("--budget",         type=int,   default=200)
    p.add_argument("--dry-run",        action="store_true")
    p.add_argument("--skip-oracle",    action="store_true")
    p.add_argument("--skip-training",  action="store_true")
    p.add_argument("--adapter-steps",  type=int,   default=1000)
    p.add_argument("--adapter-lr",     type=float, default=1e-5)
    p.add_argument("--d-inner",        type=int,   default=64)
    p.add_argument("--margin-target",  type=float, default=1.5)
    p.add_argument("--model",          default=None)
    p.add_argument("--backend",        choices=["mlx", "torch"], default="mlx")
    p.add_argument("--no-publish",     action="store_true")
    p.add_argument("--generate-problems", action="store_true")


def _run_loop(args):
    """The main autonomy loop body (extracted so main() stays clean)."""
    # ── Load problem ──────────────────────────────────────────────────────────
    problem_path = os.path.abspath(args.problem)
    problem_dir  = os.path.dirname(problem_path)
    problem      = load_problem(problem_path)

    hunt_cfg     = problem.get("hunt_config", {})
    model_name   = args.model or problem.get("model", "Qwen/Qwen3-4B-Base")
    threshold    = float(hunt_cfg.get("numerical_threshold", 0.005))
    ic_priority  = hunt_cfg.get("ic_priority", ["equal_pair"])
    checker_type = detect_checker_type(problem)
    domain_desc  = (problem.get("description") or problem.get("name") or
                    "Unknown physics domain")

    print(f"\n{'='*72}")
    print(f"  NoetherSolve Autonomy Loop")
    print(f"{'='*72}")
    print(f"  Problem:     {problem.get('name', args.problem)}")
    print(f"  Checker:     {checker_type}")
    print(f"  ICs:         {ic_priority}")
    print(f"  Threshold:   frac_var < {threshold:.1e}")
    print(f"  Oracle:      {model_name}")
    print(f"  Budget:      {args.budget} oracle calls")
    print(f"  Dry run:     {args.dry_run}")
    print(f"  Skip oracle: {args.skip_oracle}")
    print(f"  Skip train:  {args.skip_training}")
    print(f"  Publish:     {not args.no_publish}")

    # ── Show open questions queue ─────────────────────────────────────────────
    open_qs = [q for q in load_open_questions() if q.get("status") == "open"]
    if open_qs:
        print(f"\n  Open questions queue: {len(open_qs)} pending")
        for q in open_qs[:5]:
            qtype = q.get("type", "?")
            if qtype == "expression":
                print(f"    [expr] {q.get('expr', '')}  IC={q.get('ic', '?')}")
            else:
                print(f"    [dir]  {q.get('description', '')}")
        if len(open_qs) > 5:
            print(f"    ... and {len(open_qs) - 5} more (python autonomy_loop.py show-queue)")

    # ── Expand expression templates ───────────────────────────────────────────
    all_exprs = expand_expressions(hunt_cfg)

    # Also inject any open expression questions for this domain
    injected = 0
    for q in open_qs:
        if q.get("type") == "expression" and q.get("expr"):
            ic = q.get("ic", ic_priority[0])
            if ic in ic_priority or not q.get("ic"):
                if q["expr"] not in all_exprs:
                    all_exprs.append(q["expr"])
                    injected += 1

    tested = load_tested_hypotheses()
    print(f"\n  Expressions to test: {len(all_exprs)}  (injected from queue: {injected})")
    print(f"  Already in TSV:      {len(tested)}")

    # ── Phase 1: Numerical sweep ──────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  PHASE 1 — Numerical sweep  (frac_var threshold: {threshold:.1e})")
    print(f"{'─'*72}")

    numerical_passes = []
    skipped = 0

    for expr in all_exprs:
        if already_tested(expr, tested):
            skipped += 1
            continue

        passed, fv_by_ic = numerical_check(expr, checker_type, ic_priority, threshold)
        label  = expr_to_label(expr)
        fv_str = "  ".join(
            f"{ic}:{fv:.2e}" for ic, fv in fv_by_ic.items() if not np.isnan(fv)
        )
        verdict = "PASS ✓" if passed else "fail"
        print(f"  {verdict}  {label:56s}  {fv_str}")

        if passed:
            numerical_passes.append((expr, ic_priority[0], fv_by_ic))
        elif not args.dry_run and not args.skip_oracle and not args.no_publish:
            fv_note = ";".join(
                f"{ic}:{fv:.2e}" for ic, fv in fv_by_ic.items() if not np.isnan(fv)
            )
            append_tsv_row(expr, ic_priority[0], fv_by_ic,
                           None, None, "CHECKER-FAIL", fv_note)

    print(f"\n  Numerical passes:  {len(numerical_passes)}")
    print(f"  Skipped (dup):     {skipped}")
    print(f"  Fails:             {len(all_exprs) - skipped - len(numerical_passes)}")

    if args.dry_run or args.skip_oracle:
        print(f"\n  [dry-run / skip-oracle] Stopping after numerical sweep.")
        if numerical_passes:
            print(f"\n  Passes ready for oracle:")
            for expr, ic, _ in numerical_passes:
                print(f"    {expr_to_label(expr)}")
        return

    if not numerical_passes:
        print("\n  No numerical passes found. Nothing to oracle-test. Done.")
        return

    # ── Phase 2: Oracle + training loop ──────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  PHASE 2 — Oracle evaluation + adapter training")
    print(f"{'─'*72}")

    if args.backend == "mlx":
        model, tokenizer, lm_head = load_model_mlx(model_name)
    else:
        print(f"  Loading {model_name} (PyTorch)...")
        try:
            from noethersolve_torch import load_model_cpu_or_cuda
            model, tokenizer, lm_head = load_model_cpu_or_cuda(model_name)
        except ImportError:
            print("  ERROR: noethersolve_torch.py not found. Use --backend mlx.")
            return

    anthropic_client = None
    if not args.skip_training:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("\n  ⚠  ANTHROPIC_API_KEY not set — training data generation disabled.")
            print("     export ANTHROPIC_API_KEY='sk-ant-...'  or use --skip-training\n")
            args.skip_training = True
        else:
            try:
                import anthropic as _anth
                anthropic_client = _anth.Anthropic(api_key=api_key)
                print(f"  Anthropic client ready.")
            except ImportError:
                print("  ⚠  pip install anthropic")
                args.skip_training = True

    oracle_count = 0
    dual_pass    = []
    flipped      = []
    open_gaps    = []

    for expr, ic_name, fv_by_ic in numerical_passes:
        if oracle_count >= args.budget:
            print(f"\n  Budget reached ({args.budget}). Stopping.")
            break

        label  = expr_to_label(expr)
        fv_min = min(v for v in fv_by_ic.values() if not np.isnan(v))
        print(f"\n  ┌─ Candidate: {label}")
        print(f"  │  IC: {ic_name}  frac_var: {fv_min:.2e}")

        # Step A: Generate oracle question
        oracle_facts      = None
        training_examples = []

        if anthropic_client:
            print(f"  │  Generating oracle question + training data...")
            generated = call_claude_generate(
                expr, domain_desc, ic_name, fv_by_ic, threshold, anthropic_client
            )
            if generated:
                oracle_facts      = [generated["oracle_question"]]
                training_examples = generated.get("training_examples", [])
                print(f"  │  Generated: oracle question + {len(training_examples)} training examples")

        if oracle_facts is None:
            vs_relpath = problem.get("verification_set", "")
            vs_path    = (vs_relpath if os.path.isabs(vs_relpath)
                          else os.path.join(problem_dir, vs_relpath))
            if vs_path and os.path.exists(vs_path):
                with open(vs_path) as f:
                    vdata = json.load(f)
                facts_list   = vdata if isinstance(vdata, list) else vdata.get("facts", [])
                oracle_facts = facts_list
                print(f"  │  Fallback: verification set ({len(oracle_facts)} facts)")
            else:
                print(f"  │  No oracle facts — skipping")
                continue

        # Step B: Baseline oracle
        baseline = run_oracle_on_facts(model, tokenizer, oracle_facts,
                                       adapter=None, lm_head=lm_head)
        oracle_count += 1
        b_margin  = baseline["mean_margin"]
        pass_rate = f"{baseline['n_pass']}/{baseline['n_total']}"
        print(f"  │  Baseline: margin={b_margin:+.3f}  ({pass_rate} pass)")

        if b_margin > 0:
            print(f"  └─ ✓ DUAL-PASS — model already knows this!")
            dual_pass.append((expr, b_margin, fv_by_ic))
            if not args.no_publish:
                append_tsv_row(expr, ic_name, fv_by_ic, b_margin, None,
                               "DUAL-PASS", f"frac_var={fv_min:.2e}",
                               n_pass_str=pass_rate)
            continue

        print(f"  │  Oracle FAIL (margin={b_margin:+.3f})")

        if args.skip_training or not anthropic_client or not training_examples:
            open_gaps.append((expr, b_margin, fv_by_ic))
            if not args.no_publish:
                append_tsv_row(expr, ic_name, fv_by_ic, b_margin, None,
                               "ORACLE-FAIL+CHECKER-PASS",
                               f"margin={b_margin:.2f} frac_var={fv_min:.2e}",
                               n_pass_str=pass_rate)
            print(f"  └─ Recorded as ORACLE-FAIL+CHECKER-PASS")
            continue

        # Step C: Train adapter
        print(f"  │  Training adapter ({len(training_examples)} examples, "
              f"{args.adapter_steps} steps)...")
        adapter_name = (f"auto_{ic_name}_"
                        f"{datetime.datetime.now().strftime('%m%d_%H%M%S')}.npz")
        adapter_path = os.path.join(ADAPTERS_DIR, adapter_name)

        try:
            adapter = train_on_examples(
                model, tokenizer, lm_head, training_examples,
                steps=args.adapter_steps, lr=args.adapter_lr,
                d_inner=args.d_inner, margin_target=args.margin_target,
            )
            save_adapter_mlx(adapter, adapter_path)
            print(f"  │  Adapter saved: {adapter_name}")
        except Exception as exc:
            print(f"  │  Training error: {exc}")
            open_gaps.append((expr, b_margin, fv_by_ic))
            if not args.no_publish:
                append_tsv_row(expr, ic_name, fv_by_ic, b_margin, None,
                               "ORACLE-FAIL+CHECKER-PASS",
                               f"training_error frac_var={fv_min:.2e}")
            print(f"  └─ Training failed — open gap")
            continue

        # Step D: Re-evaluate
        print(f"  │  Re-evaluating with adapter...")
        repaired = run_oracle_on_facts(model, tokenizer, oracle_facts,
                                       adapter=adapter, lm_head=lm_head)
        oracle_count += 1
        r_margin   = repaired["mean_margin"]
        delta      = r_margin - b_margin
        pass_rate2 = f"{repaired['n_pass']}/{repaired['n_total']}"
        print(f"  │  Repaired: margin={r_margin:+.3f}  (Δ={delta:+.3f})  ({pass_rate2})")

        from oracle_wrapper import diagnose_quadrant
        quadrant = diagnose_quadrant(b_margin, r_margin, checker_passed=True)

        if r_margin > 0:
            print(f"  └─ 🎉 FLIPPED!  {b_margin:+.3f} → {r_margin:+.3f}")
            flipped.append((expr, b_margin, r_margin, fv_by_ic, adapter_path))
            if not args.no_publish:
                append_tsv_row(expr, ic_name, fv_by_ic, b_margin, r_margin,
                               "QUADRANT3→FLIPPED",
                               f"auto_adapter frac_var={fv_min:.2e}",
                               n_pass_str=pass_rate2)
        else:
            quad_upper = quadrant.upper()
            print(f"  └─ {quad_upper}  (margin still {r_margin:+.3f})")
            open_gaps.append((expr, b_margin, fv_by_ic))
            if not args.no_publish:
                append_tsv_row(expr, ic_name, fv_by_ic, b_margin, r_margin,
                               "ORACLE-FAIL+CHECKER-PASS",
                               f"{quad_upper} {b_margin:.2f}→{r_margin:.2f}",
                               n_pass_str=pass_rate2)

    # ── Phase 3: Generate new problems ────────────────────────────────────────
    generate_flag = getattr(args, "generate_problems", False)
    if generate_flag and anthropic_client:
        print(f"\n{'─'*72}")
        print(f"  PHASE 3 — Generating new research hypotheses")
        print(f"{'─'*72}")

        new_exprs, new_dirs = generate_new_problems(
            domain_desc, checker_type, ic_priority,
            dual_pass, flipped, open_gaps, anthropic_client,
        )

        # Convert to open_questions format
        new_qs = []
        for e in new_exprs:
            new_qs.append({
                "type":      "expression",
                "domain":    problem.get("name", "unknown"),
                "expr":      e.get("expr", ""),
                "ic":        e.get("ic", ic_priority[0]),
                "rationale": e.get("rationale", ""),
                "priority":  e.get("priority", "medium"),
            })
        for d in new_dirs:
            new_qs.append({
                "type":        "direction",
                "domain":      d.get("domain", problem.get("name", "unknown")),
                "description": d.get("description", ""),
                "priority":    d.get("priority", "medium"),
            })

        added = save_open_questions(new_qs, source="autonomy_loop")
        print(f"  Added {added} new questions to open queue.")
        print_open_questions_menu()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  NoetherSolve Autonomy Loop — Complete")
    print(f"{'='*72}")
    print(f"  Numerical passes:    {len(numerical_passes)}")
    print(f"  Oracle calls used:   {oracle_count} / {args.budget}")
    print(f"  DUAL-PASS:           {len(dual_pass)}")
    print(f"  FLIPPED:             {len(flipped)}")
    print(f"  Open gaps:           {len(open_gaps)}")

    if dual_pass:
        print(f"\n  ✓ DUAL-PASS:")
        for expr, margin, _ in dual_pass:
            print(f"    margin={margin:+.3f}  {expr_to_label(expr)}")

    if flipped:
        print(f"\n  🎉 FLIPPED:")
        for expr, b_m, r_m, _, adp in flipped:
            print(f"    {b_m:+.3f}→{r_m:+.3f}  {expr_to_label(expr)}")
            print(f"    adapter: {os.path.basename(adp)}")

    if open_gaps:
        print(f"\n  ⚠  Open gaps:")
        for expr, margin, fv_by_ic in open_gaps:
            fv_str = ";".join(
                f"{ic}:{fv:.2e}" for ic, fv in fv_by_ic.items() if not np.isnan(fv)
            )
            print(f"    margin={margin:+.3f}  {expr_to_label(expr)}  {fv_str}")

    if not args.no_publish:
        print(f"\n  Results: {CANDIDATES_TSV}")
        open_count = len([q for q in load_open_questions() if q.get("status") == "open"])
        if open_count:
            print(f"  Open queue: {open_count} questions  "
                  f"(python autonomy_loop.py show-queue)")
    print()


if __name__ == "__main__":
    main()
