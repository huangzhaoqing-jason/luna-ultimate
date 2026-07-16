#!/usr/bin/env python3
"""启动 Luna native serve（完整异构轨）。

用法：
  python3 scripts/run_serve.py --preset tiny --device auto --port 11435
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description="启动 Luna serve（Native 轨）")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=11435)
    p.add_argument("--preset", default="tiny")
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mlx", "cann", "dml", "cpu"],
    )
    args = p.parse_args()
    from serve.luna_server import serve
    serve(args.host, args.port, args.preset, args.device)


if __name__ == "__main__":
    main()
