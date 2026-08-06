# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small git helpers used by the bisection harness."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git(repo_root: Path, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command in ``repo_root``."""
    result = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"git {' '.join(args)} failed")
    return result


def resolve_ref(repo_root: Path, ref: str) -> str:
    """Resolve a git ref to a full commit SHA."""
    return git(repo_root, ["rev-parse", "--verify", f"{ref}^{{commit}}"]).stdout.strip()


def is_ancestor(repo_root: Path, ancestor_sha: str, descendant_sha: str) -> bool:
    """Return True if ``ancestor_sha`` is an ancestor of ``descendant_sha``."""
    return git(repo_root, ["merge-base", "--is-ancestor", ancestor_sha, descendant_sha], check=False).returncode == 0


def candidate_commits(repo_root: Path, good_sha: str, bad_sha: str) -> list[str]:
    """Return candidate commits on the ancestry path from good(exclusive) to bad(inclusive)."""
    if not is_ancestor(repo_root, good_sha, bad_sha):
        raise RuntimeError(f"Known-good commit {good_sha[:12]} is not an ancestor of bad commit {bad_sha[:12]}.")
    result = git(repo_root, ["rev-list", "--ancestry-path", "--reverse", f"{good_sha}..{bad_sha}"])
    commits = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not commits or commits[-1] != bad_sha:
        commits.append(bad_sha)
    return commits


def commit_summary(repo_root: Path, commit_sha: str) -> dict[str, str]:
    """Return basic metadata for a commit."""
    fmt = "%H%x00%P%x00%an%x00%ae%x00%ad%x00%s"
    result = git(repo_root, ["show", "-s", f"--format={fmt}", "--date=iso-strict", commit_sha])
    commit, parents, author, email, date, subject = result.stdout.rstrip("\n").split("\x00", 5)
    return {
        "commit_sha": commit,
        "parents": parents,
        "author": author,
        "author_email": email,
        "date": date,
        "subject": subject,
    }


def diff_name_status(repo_root: Path, base_sha: str, target_sha: str) -> list[dict[str, str]]:
    """Return changed files between two commits using git name-status."""
    result = git(repo_root, ["diff", "--name-status", base_sha, target_sha])
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            rows.append({"status": parts[0], "path": parts[-1]})
    return rows


def diff_stat(repo_root: Path, base_sha: str, target_sha: str) -> str:
    """Return a compact git diffstat."""
    return git(repo_root, ["diff", "--stat", base_sha, target_sha]).stdout.strip()
