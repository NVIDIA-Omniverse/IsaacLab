# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""JSON/file helpers for the bisection harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from ``path``."""
    with path.open(encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a pretty JSON object to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
