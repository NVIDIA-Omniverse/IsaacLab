#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Write an initial diagnosis report for a completed bisection run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_GATE_DIR = Path(__file__).parent
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

from bisection.diagnosis import write_diagnosis  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose the first bad commit from a bisection run.")
    parser.add_argument("--run_dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    diagnosis = write_diagnosis(args.run_dir)
    first_bad = diagnosis["first_bad_commit"]["commit_sha"][:12]
    print(f"[diagnose_bad_commit] wrote diagnosis for {first_bad} -> {args.run_dir / 'diagnosis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
