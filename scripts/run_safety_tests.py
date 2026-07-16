#!/usr/bin/env python3
"""Run the Luna safety test suite and report.

Usage:
  python scripts/run_safety_tests.py
  python scripts/run_safety_tests.py --strict   # exit non-zero on any failure
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety.tests import SafetyTestSuite, assert_passes
from safety.charter import verify_charter, verify_install, write_install_hash
from safety.audit import AuditLog


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strict", action="store_true")
    p.add_argument("--json", type=str, default=None)
    p.add_argument("--base_dir", type=str, default=".luna")
    args = p.parse_args()

    base = (ROOT / args.base_dir).resolve()
    # Boot: verify charter (stamp install hash on first run)
    try:
        verify_install(str(base))
    except Exception as e:
        print(f"CHARTER TAMPER: {e}", file=sys.stderr)
        sys.exit(2)

    suite = SafetyTestSuite()
    results = suite.run()
    passed = sum(1 for r in results if r.passed)
    print(suite.summary())
    print()
    print(suite.values_summary())
    values = suite.run_values()
    values_passed = sum(1 for r in values if r.passed)
    print(f"\nCharter hash: {verify_charter()[:16]}…")
    try:
        from safety.values import verify_values
        print(f"Values hash:  {verify_values()[:16]}…")
    except Exception as e:
        print(f"Values hash:  ERROR {e}")
        values_passed = 0

    audit = AuditLog(str(base / "audit.log"))
    print(f"Audit chain valid: {audit.verify_chain()}  entries: {len(audit.entries)}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({
                "passed": passed,
                "total": len(results),
                "results": [asdict(r) for r in results],
                "values_passed": values_passed,
                "values_total": len(values),
                "values_results": [asdict(r) for r in values],
                "charter_hash": verify_charter(),
                "audit_chain_valid": audit.verify_chain(),
            }, f, indent=2)

    if args.strict and (passed != len(results) or values_passed != len(values)):
        sys.exit(1)


if __name__ == "__main__":
    main()
