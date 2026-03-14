#!/usr/bin/env python3
"""
Claim/release problems in Operation Frontier.

Coordination protocol adapted from autoresearch-at-home (mutable-state-inc).
Prevents duplicate work when multiple contributors hunt the same domain.

Usage:
    python claim.py list                          # show active claims
    python claim.py claim --problem <name> --expr "<expr>" --handle <you>
    python claim.py release --id <claim_id>
    python claim.py gc                            # remove expired claims
"""

import argparse, hashlib, json, os, sys
from datetime import datetime, timezone, timedelta

HERE        = os.path.dirname(os.path.abspath(__file__))
CLAIMS_FILE = os.path.join(HERE, "claims.json")
CANDIDATES  = os.path.join(HERE, "results", "candidates.tsv")
EXPIRY_HOURS = 4


def load_claims():
    with open(CLAIMS_FILE) as f:
        return json.load(f)

def save_claims(data):
    with open(CLAIMS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  claims.json updated ({len(data['active_claims'])} active claims)")

def now_utc():
    return datetime.now(timezone.utc)

def parse_dt(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def is_expired(claim):
    return now_utc() > parse_dt(claim["expires_at"])

def make_id(problem, expr, handle):
    raw = f"{problem}:{expr}:{handle}:{now_utc().isoformat()}"
    return hashlib.sha1(raw.encode()).hexdigest()[:8]


def cmd_list(args):
    data = load_claims()
    gc_expired(data, silent=True)
    claims = data["active_claims"]
    if not claims:
        print("  No active claims. Domain is open — start with: python claim.py claim ...")
        return
    print(f"  {'ID':8s}  {'problem':35s}  {'claimer':15s}  {'expires':20s}  expression")
    print("  " + "-" * 110)
    for c in claims:
        exp = parse_dt(c["expires_at"])
        remaining = exp - now_utc()
        hours_left = remaining.total_seconds() / 3600
        tag = f"{hours_left:.1f}h" if hours_left > 0 else "EXPIRED"
        print(f"  {c['id']:8s}  {c['problem']:35s}  {c['claimer']:15s}  {tag:20s}  {c['expression']}")


def cmd_claim(args):
    data   = load_claims()
    gc_expired(data, silent=True)

    # Check for semantic near-duplicate in claims
    expr_norm = args.expr.replace(" ", "").lower()
    for c in data["active_claims"]:
        if c["expression"].replace(" ", "").lower() == expr_norm:
            print(f"  DUPLICATE CLAIM: '{args.expr}' already claimed by {c['claimer']} "
                  f"(id={c['id']}, expires {c['expires_at']})")
            sys.exit(1)

    # Check candidates.tsv for already-closed holes
    if os.path.exists(CANDIDATES):
        with open(CANDIDATES) as f:
            for line in f:
                if expr_norm in line.replace(" ", "").lower():
                    print(f"  WARNING: Similar expression found in candidates.tsv:")
                    print(f"    {line.strip()}")
                    resp = input("  Continue anyway? [y/N] ").strip().lower()
                    if resp != "y":
                        sys.exit(0)

    t_now    = now_utc()
    t_expire = t_now + timedelta(hours=EXPIRY_HOURS)
    claim = {
        "id":          make_id(args.problem, args.expr, args.handle),
        "problem":     args.problem,
        "expression":  args.expr,
        "claimer":     args.handle,
        "claimed_at":  t_now.isoformat(),
        "expires_at":  t_expire.isoformat(),
        "status":      "active",
    }
    data["active_claims"].append(claim)
    save_claims(data)
    print(f"\n  Claimed: id={claim['id']}")
    print(f"  Problem:    {args.problem}")
    print(f"  Expression: {args.expr}")
    print(f"  Expires:    {t_expire.strftime('%Y-%m-%d %H:%M UTC')} ({EXPIRY_HOURS}h)")
    print(f"\n  Run your checker, then publish with:")
    print(f"    python claim.py release --id {claim['id']}")


def cmd_release(args):
    data   = load_claims()
    before = len(data["active_claims"])
    data["active_claims"] = [c for c in data["active_claims"] if c["id"] != args.id]
    if len(data["active_claims"]) == before:
        print(f"  Claim id={args.id} not found.")
        sys.exit(1)
    save_claims(data)
    print(f"  Released claim id={args.id}. Don't forget to PR your results to candidates.tsv!")


def gc_expired(data, silent=False):
    before = len(data["active_claims"])
    data["active_claims"] = [c for c in data["active_claims"] if not is_expired(c)]
    removed = before - len(data["active_claims"])
    if removed and not silent:
        print(f"  GC: removed {removed} expired claim(s)")
    return removed


def cmd_gc(args):
    data = load_claims()
    removed = gc_expired(data)
    save_claims(data)
    if not removed:
        print("  No expired claims to remove.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Operation Frontier claim manager")
    sub    = parser.add_subparsers(dest="cmd")

    sub.add_parser("list",  help="Show active claims")
    sub.add_parser("gc",    help="Remove expired claims")

    p_claim = sub.add_parser("claim", help="Claim a problem before running")
    p_claim.add_argument("--problem", required=True, help="Problem name (matches problems/*.yaml)")
    p_claim.add_argument("--expr",    required=True, help="Expression you're testing")
    p_claim.add_argument("--handle",  required=True, help="Your GitHub handle or name")

    p_rel = sub.add_parser("release", help="Release a claim after publishing")
    p_rel.add_argument("--id", required=True, help="Claim ID to release")

    args = parser.parse_args()
    if   args.cmd == "list":    cmd_list(args)
    elif args.cmd == "claim":   cmd_claim(args)
    elif args.cmd == "release": cmd_release(args)
    elif args.cmd == "gc":      cmd_gc(args)
    else:
        parser.print_help()
