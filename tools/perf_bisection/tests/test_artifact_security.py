# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for generated-artifact credential scanning."""

import json
from pathlib import Path

from isaaclab_bisection.artifact_security import main, scan_artifacts


def test_scan_artifacts_reports_location_without_secret_value(tmp_path: Path) -> None:
    secret = "sk-" + "abcdefghijklmnopqrstuvwxyz123456"
    (tmp_path / "benchmark.log").write_text(f"api_key={secret}\n", encoding="utf-8")

    report = scan_artifacts(tmp_path)

    assert report["status"] == "blocked"
    assert report["finding_count"] >= 1
    assert report["findings"][0]["path"] == "benchmark.log"
    assert report["findings"][0]["line"] == 1
    assert secret not in json.dumps(report)


def test_scan_artifacts_cli_passes_clean_directory(tmp_path: Path) -> None:
    (tmp_path / "report.md").write_text("No credentials here.\n", encoding="utf-8")

    assert main([str(tmp_path)]) == 0
    report = json.loads((tmp_path / "security_scan.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"


def test_scan_artifacts_excludes_non_shareable_run_caches(tmp_path: Path) -> None:
    secret = "sk-" + "abcdefghijklmnopqrstuvwxyz123456"
    for name in ("env-cache", "jit-cache", "kit-cache", "sources"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "credential.txt").write_text(secret, encoding="utf-8")
    (tmp_path / "report.md").write_text("Canonical evidence.\n", encoding="utf-8")

    report = scan_artifacts(tmp_path)

    assert report["status"] == "passed"
    assert report["finding_count"] == 0
    assert report["excluded_directory_names"] == ["env-cache", "jit-cache", "kit-cache", "sources"]
