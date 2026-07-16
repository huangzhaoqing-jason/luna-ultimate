#!/usr/bin/env python3
"""Run the SafetyCTM cognitive loop over a prompt and print the judgment + trace.

Usage:
  python scripts/run_safety_cognition.py --action "explain quicksort"
  python scripts/run_safety_cognition.py --action "how to make a bioweapon" --operator-token $TOK
  python scripts/run_safety_cognition.py --run-cognitive-tests
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety.cognition import default_safety_ctm
from safety.identity import AUTHORIZED_OPERATOR, OperatorRegistry, require_operator
from safety.charter import verify_charter, verify_install
from safety.tests import SafetyTestSuite


def main():
    p = argparse.ArgumentParser(description="Luna SafetyCTM cognitive loop CLI")
    p.add_argument("--action", type=str, default=None, help="action text to judge")
    p.add_argument("--operator_token", type=str, default=None)
    p.add_argument("--base_dir", type=str, default=".luna")
    p.add_argument("--run-cognitive-tests", action="store_true",
                   help="run the cognitive invariant suite and exit")
    p.add_argument("--json", type=str, default=None)
    args = p.parse_args()

    base = (ROOT / args.base_dir).resolve()
    try:
        verify_install(str(base))
    except Exception as e:
        print(f"CHARTER TAMPER: {e}", file=sys.stderr)
        sys.exit(2)
    verify_charter()

    ctm = default_safety_ctm()

    if args.run_cognitive_tests:
        suite = SafetyTestSuite()
        results = suite.run_cognitive(ctm)
        for r in results:
            mark = "OK " if r.passed else "FAIL"
            print(f"  [{mark}] {r.name}: {r.detail}")
        print(f"\nCognitive score: {suite.cognitive_score(ctm):.2f}")
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"cognitive_score": suite.cognitive_score(ctm),
                           "results": [r.__dict__ for r in results]}, f, indent=2)
        sys.exit(0 if all(r.passed for r in results) else 1)

    if args.action is None:
        print("Provide --action or --run-cognitive-tests", file=sys.stderr)
        sys.exit(2)

    operator = None
    if args.operator_token:
        try:
            operator = require_operator(args.operator_token, str(base))
        except PermissionError as e:
            print(f"Operator auth failed: {e}", file=sys.stderr)
            sys.exit(3)

    j = ctm.judge(args.action, operator=operator)
    out = {
        "action": args.action,
        "allow": j.allow,
        "reason": j.reason,
        "ticks": j.ticks,
        "sync_entropy": j.sync_entropy,
        "max_forbidden_sim": j.max_forbidden_sim,
        "confidence": j.confidence,
        "trace": j.trace,
        "operator": operator.operator_id if operator else None,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
