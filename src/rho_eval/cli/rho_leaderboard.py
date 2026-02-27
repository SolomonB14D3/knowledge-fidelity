"""CLI entry point for Fidelity-Bench leaderboard management.

Usage:
    rho-leaderboard show                             # Display local leaderboard
    rho-leaderboard show --sort truth_gap            # Sort by truth-gap
    rho-leaderboard submit cert.json                 # Add certificate to leaderboard
    rho-leaderboard compare model-a model-b          # Side-by-side comparison
    rho-leaderboard push --repo rho-eval/bench       # Push to HF Hub
    rho-leaderboard pull --repo rho-eval/bench       # Pull from HF Hub
"""

import argparse
import json
import sys
from pathlib import Path


DEFAULT_LEADERBOARD = Path("leaderboard.json")


def cmd_show(args):
    """Display the local leaderboard."""
    from rho_eval.benchmarking.leaderboard import Leaderboard

    lb_path = Path(args.file)
    if not lb_path.exists():
        print(f"No leaderboard found at {lb_path}")
        print("Run 'rho-leaderboard submit <cert.json>' to create one.")
        return

    lb = Leaderboard.load(lb_path)
    print(f"\nFidelity-Bench Leaderboard ({len(lb)} entries)")
    print(f"Source: {lb_path}\n")

    try:
        ranked = lb.rank(metric=args.sort)
        # Simple table output
        print(f"{'#':<4} {'Model':<40} {'Grade':<6} {'Composite':<10} {'Truth-Gap':<10}")
        print("─" * 74)
        for i, entry in enumerate(ranked, 1):
            print(
                f"{i:<4} {entry.model[:38]:<40} {entry.fidelity_grade:<6} "
                f"{entry.composite_score:<10.4f} {entry.truth_gap:<10.4f}"
            )
    except NotImplementedError:
        print("⚠  Leaderboard ranking not yet implemented.")
        print(f"   {len(lb)} entries loaded from {lb_path}")


def cmd_submit(args):
    """Submit a FidelityCertificate to the leaderboard."""
    from rho_eval.benchmarking.schema import FidelityCertificate
    from rho_eval.benchmarking.leaderboard import Leaderboard

    cert_path = Path(args.cert)
    if not cert_path.exists():
        print(f"Certificate not found: {cert_path}")
        sys.exit(1)

    # Load certificate
    cert_data = json.loads(cert_path.read_text())
    # TODO: Deserialize into FidelityCertificate
    print(f"Loaded certificate from {cert_path}")

    # Load or create leaderboard
    lb_path = Path(args.file)
    if lb_path.exists():
        lb = Leaderboard.load(lb_path)
    else:
        lb = Leaderboard()

    try:
        # entry = lb.submit(cert)
        # lb.save(lb_path)
        # print(f"Added {entry.model} (grade={entry.fidelity_grade}) to leaderboard")
        # print(f"Leaderboard saved to {lb_path} ({len(lb)} entries)")
        print("⚠  Leaderboard submission not yet implemented.")
    except NotImplementedError:
        print("⚠  Leaderboard submission not yet implemented.")


def cmd_compare(args):
    """Compare two models side-by-side."""
    from rho_eval.benchmarking.leaderboard import Leaderboard

    lb_path = Path(args.file)
    if not lb_path.exists():
        print(f"No leaderboard found at {lb_path}")
        sys.exit(1)

    lb = Leaderboard.load(lb_path)
    try:
        comparison = lb.compare(args.model_a, args.model_b)
        print(comparison)
    except NotImplementedError:
        print("⚠  Leaderboard comparison not yet implemented.")


def cmd_push(args):
    """Push leaderboard to HuggingFace Hub."""
    from rho_eval.benchmarking.leaderboard import Leaderboard

    lb_path = Path(args.file)
    if not lb_path.exists():
        print(f"No leaderboard found at {lb_path}")
        sys.exit(1)

    lb = Leaderboard.load(lb_path)
    try:
        url = lb.push_to_hub(args.repo, token=args.token)
        print(f"Pushed {len(lb)} entries to {url}")
    except NotImplementedError:
        print("⚠  HF Hub push not yet implemented.")
        print("   Requires: pip install huggingface-hub")


def cmd_pull(args):
    """Pull leaderboard from HuggingFace Hub."""
    from rho_eval.benchmarking.leaderboard import Leaderboard

    lb = Leaderboard()
    try:
        lb.pull_from_hub(args.repo, token=args.token)
        lb_path = Path(args.file)
        lb.save(lb_path)
        print(f"Pulled {len(lb)} entries from {args.repo}")
        print(f"Saved to {lb_path}")
    except NotImplementedError:
        print("⚠  HF Hub pull not yet implemented.")
        print("   Requires: pip install huggingface-hub")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="rho-leaderboard",
        description="Fidelity-Bench Leaderboard Management",
    )
    parser.add_argument(
        "--file", "-f", type=str, default=str(DEFAULT_LEADERBOARD),
        help=f"Leaderboard JSON file (default: {DEFAULT_LEADERBOARD})",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── show ──────────────────────────────────────────────────────────────
    p_show = subparsers.add_parser("show", help="Display the leaderboard")
    p_show.add_argument(
        "--sort", type=str, default="composite",
        help="Sort metric: composite, truth_gap, grade (default: composite)",
    )

    # ── submit ────────────────────────────────────────────────────────────
    p_submit = subparsers.add_parser("submit", help="Submit a certificate")
    p_submit.add_argument("cert", help="Path to FidelityCertificate JSON")

    # ── compare ───────────────────────────────────────────────────────────
    p_compare = subparsers.add_parser("compare", help="Compare two models")
    p_compare.add_argument("model_a", help="First model name")
    p_compare.add_argument("model_b", help="Second model name")

    # ── push ──────────────────────────────────────────────────────────────
    p_push = subparsers.add_parser("push", help="Push to HuggingFace Hub")
    p_push.add_argument("--repo", required=True, help="HF Hub dataset ID")
    p_push.add_argument("--token", type=str, default=None, help="HF API token")

    # ── pull ──────────────────────────────────────────────────────────────
    p_pull = subparsers.add_parser("pull", help="Pull from HuggingFace Hub")
    p_pull.add_argument("--repo", required=True, help="HF Hub dataset ID")
    p_pull.add_argument("--token", type=str, default=None, help="HF API token")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "show": cmd_show,
        "submit": cmd_submit,
        "compare": cmd_compare,
        "push": cmd_push,
        "pull": cmd_pull,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
