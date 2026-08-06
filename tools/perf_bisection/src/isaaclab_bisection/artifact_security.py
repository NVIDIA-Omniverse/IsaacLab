# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scan generated bisection artifacts for high-confidence credential patterns."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_MAX_FILE_SIZE = 10 * 1024 * 1024
_PATTERNS = (
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("authorization_bearer", re.compile(r"(?i)\bauthorization\s*:\s*bearer\s+\S+")),
    ("credentialed_url", re.compile(r"(?i)\bhttps?://[^/\s:@]+:[^/\s@]+@")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")),
    ("openai_style_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    (
        "credential_assignment",
        re.compile(r"(?i)\b(?:api[_-]?key|token|password|passwd|secret)\s*[=:]\s*[\"']?[A-Za-z0-9_./+=-]{12,}"),
    ),
)


def scan_artifacts(root: Path) -> dict[str, Any]:
    """Return a value-free report of likely credentials below ``root``."""
    findings: list[dict[str, Any]] = []
    for path in _iter_text_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for kind, pattern in _PATTERNS:
                if pattern.search(line):
                    findings.append(
                        {
                            "kind": kind,
                            "path": str(path.relative_to(root)),
                            "line": line_number,
                        }
                    )
    return {
        "schema_version": 1,
        "status": "blocked" if findings else "passed",
        "finding_count": len(findings),
        "findings": findings,
        "note": "Potential secret values are intentionally omitted. Review findings before sharing artifacts.",
    }


def _iter_text_files(root: Path) -> Iterator[Path]:
    """Yield bounded regular files while excluding this scanner's own report."""
    for path in root.rglob("*"):
        if path.name == "security_scan.json" or not path.is_file():
            continue
        try:
            if path.stat().st_size <= _MAX_FILE_SIZE:
                yield path
        except OSError:
            continue


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Scan an artifact directory and return nonzero when sharing must stop."""
    args = _parse_args(argv)
    report = scan_artifacts(args.artifact_dir)
    output = args.output or args.artifact_dir / "security_scan.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Artifact security scan: {report['status']} ({report['finding_count']} finding(s)); report={output}")
    return 1 if report["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
