#!/usr/bin/env python3
"""Enroll the authorized operator (黄照清) and print a one-time token.

The token is shown ONCE. Store it securely; only its salted hash is kept.

Usage:
  python scripts/enroll_operator.py
  python scripts/enroll_operator.py --reissue
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety.identity import AUTHORIZED_OPERATOR, OperatorRegistry
from safety.charter import verify_charter, verify_install


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reissue", action="store_true", help="Reissue a fresh token")
    p.add_argument("--base_dir", type=str, default=".luna")
    args = p.parse_args()

    base = (ROOT / args.base_dir).resolve()
    verify_install(str(base))
    verify_charter()

    reg = OperatorRegistry(base_dir=str(base))
    print(f"Authorized operator: {AUTHORIZED_OPERATOR.name} "
          f"(DOB {AUTHORIZED_OPERATOR.dob}, {AUTHORIZED_OPERATOR.nationality})")
    print(f"Operator id: {AUTHORIZED_OPERATOR.operator_id}")
    token = reg.bootstrap()
    print("\nONE-TIME operator token (store securely, shown once):")
    print(token)
    print("\nThe system will now require this token for sensitive actions.")
    print("Loyalty model: priority service for legitimate goals. The safety")
    print("charter still binds the operator — harmful commands are refused.")


if __name__ == "__main__":
    main()
