# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check whether candidate IsaacLab APIs satisfy the pinned perf-smoke driver."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

TOOLING_INCOMPATIBLE_EXIT_CODE = 86


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def check_capabilities() -> list[str]:
    """Return missing API descriptions required by ``perf_runtime.py``."""
    missing: list[str] = []
    checks = (
        ("isaaclab.app", ("AppLauncher", "launch_simulation")),
        (
            "isaaclab.test.benchmark",
            ("BaseIsaacLabBenchmark", "BenchmarkMonitor", "builders", "capture", "stepping"),
        ),
        ("isaaclab.test.benchmark.schema", ("StartupTime",)),
        ("isaaclab_tasks.utils", ("setup_preset_cli", "resolve_task_config")),
    )
    for module_name, names in checks:
        try:
            module = __import__(module_name, fromlist=list(names))
        except Exception as exc:  # noqa: BLE001 - the compatibility report records the exact import failure
            missing.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        for name in names:
            if not hasattr(module, name):
                missing.append(f"{module_name}.{name}")
    return missing


def main() -> int:
    args = _parse_args()
    payload: dict[str, object]
    try:
        missing = check_capabilities()
        payload = {
            "status": "compatible" if not missing else "incompatible",
            "missing": missing,
        }
    except Exception as exc:  # pragma: no cover - final guard for unusual import machinery
        payload = {
            "status": "incompatible",
            "missing": [f"{type(exc).__name__}: {exc}"],
            "traceback": traceback.format_exc(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if payload["status"] != "compatible":
        print("PERF_SMOKE_TOOLING_INCOMPATIBLE", json.dumps(payload["missing"]), flush=True)
        return TOOLING_INCOMPATIBLE_EXIT_CODE
    print("PERF_SMOKE_TOOLING_COMPATIBLE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
