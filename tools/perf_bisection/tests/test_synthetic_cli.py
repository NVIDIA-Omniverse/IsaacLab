# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installed-CLI smoke test using a self-contained synthetic target repository."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit(repo: Path, message: str, value: str) -> str:
    (repo / "candidate.txt").write_text(value, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def test_synthetic_bisection_runs_against_external_target_repo(tmp_path: Path) -> None:
    repo = tmp_path / "target"
    tooling = repo / "tools" / "perf_smoke_test"
    tooling.mkdir(parents=True)
    (tooling / "contract.txt").write_text("synthetic tooling\n", encoding="utf-8")
    (tooling / "dev").mkdir()
    (tooling / "dev" / "stub_benchmark.py").write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=Path, required=True)
parser.add_argument("--fps_mean", type=float, required=True)
args, _ = parser.parse_known_args()
args.out_dir.mkdir(parents=True, exist_ok=True)
(args.out_dir / "synthetic.json").write_text(json.dumps({"fps": args.fps_mean}), encoding="utf-8")
""".lstrip(),
        encoding="utf-8",
    )
    (tooling / "build_bench_result.py").write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--task_id", required=True)
parser.add_argument("--physics_backend", required=True)
parser.add_argument("--render_backend", default="")
parser.add_argument("--artifact_dir", type=Path, required=True)
args, _ = parser.parse_known_args()
launch = json.loads((args.artifact_dir / "launch_config.json").read_text(encoding="utf-8"))
sample = json.loads((args.artifact_dir / "synthetic.json").read_text(encoding="utf-8"))
result = {
    "task_id": args.task_id,
    "backend": launch["backend_key"],
    "physics_backend": args.physics_backend,
    "render_backend": args.render_backend or None,
    "backend_key": launch["backend_key"],
    "preset": launch["preset"],
    "perf_smoke_test_info_present": True,
    "raw_fps_mean": sample["fps"],
    "launch_config_hash": launch["launch_config_hash"],
    "benchmark_contract_hash": launch["benchmark_contract_hash"],
    "schema_version": "1.0",
}
(args.artifact_dir / "perf_smoke_test_result.json").write_text(json.dumps(result), encoding="utf-8")
""".lstrip(),
        encoding="utf-8",
    )
    _git(repo, "init", "-b", "main")
    good_sha = _commit(repo, "good", "good\n")
    first_bad_sha = _commit(repo, "regress", "bad\n")
    bad_sha = _commit(repo, "later", "still bad\n")
    output = tmp_path / "output"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "isaaclab_bisection.cli",
            "bisect-range",
            "--repo_root",
            str(repo),
            "--work_dir",
            str(output),
            "--runner_mode",
            "synthetic",
            "--good_ref",
            good_sha,
            "--bad_ref",
            bad_sha,
            "--tooling_ref",
            bad_sha,
            "--task_id",
            "Synthetic-Task",
            "--backend_key",
            "physx",
            "--num_envs",
            "1",
            "--synthetic_first_bad_ref",
            first_bad_sha,
            "--reference_runs",
            "1",
            "--max_reference_runs",
            "1",
            "--candidate_runs",
            "1",
            "--max_candidate_runs",
            "1",
            "--warmup_runs",
            "0",
            "--progress",
            "quiet",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["suspected_first_bad_commit"] == first_bad_sha


def test_real_runner_requires_explicit_target_trust_confirmation(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "task_id": "Isaac-Cartpole-Direct",
                "backend_key": "physx",
                "good_ref": "good",
                "bad_ref": "bad",
                "gpu_model": "L40S",
                "runner": {"mode": "docker-reconstruct", "image": "image:tag"},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "isaaclab_bisection.cli",
            "benchmark-commit",
            "--plan",
            str(plan),
            "--commit",
            "candidate",
            "--repo_root",
            str(tmp_path / "target"),
            "--work_dir",
            str(tmp_path / "output"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "SECURITY_BLOCKED=real runner modes require --trust_target_code" in result.stderr
