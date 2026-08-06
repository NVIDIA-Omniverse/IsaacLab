# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for separating the installed agent from the target repository."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab_bisection import cli, runner
from isaaclab_bisection.bisection import engine
from isaaclab_bisection.bisection.models import BisectionPlan, RunnerSpec


def _plan() -> BisectionPlan:
    return BisectionPlan(
        task_id="Task-v0",
        backend_key="physx",
        good_ref="good",
        bad_ref="bad",
        gpu_model="test-gpu",
        runner=RunnerSpec(mode="synthetic"),
    )


def test_workflow_cli_accepts_external_repo_root(tmp_path: Path, monkeypatch) -> None:
    target_repo = tmp_path / "external-isaaclab"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "isaaclab-bisect",
            "dry-run",
            "--repo_root",
            str(target_repo),
            "--plan",
            "plan.json",
            "--output_dir",
            "out",
        ],
    )

    args = cli._parse_args()

    assert args.repo_root == target_repo


def test_candidate_resolution_uses_external_repo_root(tmp_path: Path, monkeypatch) -> None:
    target_repo = tmp_path / "external-isaaclab"
    calls: list[tuple[str, Path]] = []

    def resolve(repo_root: Path, ref: str) -> str:
        calls.append((ref, repo_root))
        return f"{ref}-sha"

    def candidates(repo_root: Path, good_sha: str, bad_sha: str) -> list[str]:
        calls.append((f"{good_sha}..{bad_sha}", repo_root))
        return [bad_sha]

    monkeypatch.setattr(engine, "resolve_ref", resolve)
    monkeypatch.setattr(engine, "candidate_commits", candidates)

    payload = engine.build_candidates(_plan(), target_repo)

    assert payload["candidates"] == ["bad-sha"]
    assert calls == [
        ("good", target_repo.resolve()),
        ("bad", target_repo.resolve()),
        ("good-sha..bad-sha", target_repo.resolve()),
    ]


def test_engine_invokes_package_runner_with_target_repo(tmp_path: Path) -> None:
    target_repo = tmp_path / "external-isaaclab"

    command = engine.format_runner_command(
        _plan(),
        tmp_path / "output",
        "abc123",
        tmp_path / "output" / "artifacts",
        repo_root=target_repo,
    )

    assert command[:3] == [sys.executable, "-m", "isaaclab_bisection.runner"]
    assert command[command.index("--repo_root") + 1] == str(target_repo.resolve())
    assert "bisect_single_commit_runner.py" not in " ".join(command)


def test_source_clone_uses_external_repo_root(tmp_path: Path, monkeypatch) -> None:
    target_repo = tmp_path / "external-isaaclab"
    source_dir = tmp_path / "candidate"
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs):
        commands.append(command)
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(runner, "_run", run)
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: argparse.Namespace(returncode=0))

    runner._materialize_source_clone(source_dir, "abc123", target_repo)

    assert commands[0] == [
        "git",
        "clone",
        "--no-checkout",
        str(target_repo.resolve()),
        str(source_dir),
    ]
    assert commands[1][3] == str(target_repo.resolve())
