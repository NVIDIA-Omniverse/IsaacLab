"""
infra/commits.py — Commit enumeration and diff fetching.

Supports two backends:
  - git (default): subprocess git commands against a local repo
  - GitHub API: urllib.request calls against the GitHub REST API (no extra deps)

See DESIGN.md Section 3.6 for the authoritative interface contract.
"""

from __future__ import annotations

import json
import subprocess
import urllib.request
import urllib.error
from pathlib import Path


# ---------------------------------------------------------------------------
# Dependency-file detection
# ---------------------------------------------------------------------------

_DEP_FILE_NAMES = {
    "requirements.txt",
    "setup.cfg",
    "setup.py",
    "pyproject.toml",
}

_DEP_DIR_PREFIX = "requirements/"


def _is_dep_file(path: str) -> bool:
    """Return True if *path* is a dependency-related file.

    Matches:
      - requirements.txt
      - setup.cfg
      - setup.py
      - pyproject.toml
      - requirements/<anything>   (any file inside a requirements/ directory)

    The check is intentionally path-agnostic: it looks at the basename and at
    any "requirements/" directory component, so it works for both root-level
    and nested occurrences (e.g. source/requirements.txt).
    """
    p = Path(path)
    if p.name in _DEP_FILE_NAMES:
        return True
    # Check whether any component of the path is a "requirements" directory
    # containing this file (e.g. "requirements/gpu.txt").
    parts = p.parts
    for i, part in enumerate(parts[:-1]):
        if part == "requirements":
            return True
    # Also catch "requirements/" prefix literally (forward-slash paths)
    if path.startswith(_DEP_DIR_PREFIX) or f"/{_DEP_DIR_PREFIX}" in path:
        return True
    return False


# ---------------------------------------------------------------------------
# Git backend helpers
# ---------------------------------------------------------------------------

def _git(args: list[str], *, repo_path: Path) -> str:
    """Run a git command inside *repo_path* and return stdout as a string.

    Raises RuntimeError on non-zero exit.
    """
    cmd = ["git", "-C", str(repo_path)] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git command failed (exit {result.returncode}): {' '.join(cmd)}\n"
            f"stderr: {result.stderr.strip()}"
        )
    return result.stdout


def _git_dep_files_changed(sha: str, *, repo_path: Path) -> bool:
    """Return True if the commit *sha* touched any dependency file."""
    # git diff-tree lists files changed in a single commit relative to its
    # (first) parent.  --no-commit-id suppresses the commit SHA prefix line.
    output = _git(
        ["diff-tree", "--no-commit-id", "-r", "--name-only", sha],
        repo_path=repo_path,
    )
    for line in output.splitlines():
        line = line.strip()
        if line and _is_dep_file(line):
            return True
    return False


# ---------------------------------------------------------------------------
# GitHub API backend helpers
# ---------------------------------------------------------------------------

def _github_request(url: str, *, token: str | None) -> dict | list:
    """Fetch *url* from the GitHub API and return the parsed JSON body."""
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"GitHub API error {exc.code} for {url}: {body}"
        ) from exc


def _github_dep_files_changed(commit_data: dict) -> bool:
    """Return True if the GitHub commit object contains any dep file."""
    for f in commit_data.get("files", []):
        if _is_dep_file(f.get("filename", "")):
            return True
    return False


# ---------------------------------------------------------------------------
# enumerate_commits
# ---------------------------------------------------------------------------

def enumerate_commits(
    good_sha: str,
    bad_sha: str,
    *,
    repo_path: Path | None = None,
    github_repo: str | None = None,
    github_token: str | None = None,
) -> list[dict]:
    """Return commits in range (good_sha, bad_sha] ordered oldest-first.

    Each dict has the keys:
        sha              (str) — full 40-char SHA
        short_sha        (str) — first 7 chars
        date             (str) — ISO-8601 datetime string
        author           (str) — author name
        message          (str) — first line of commit message (subject)
        dep_files_changed (bool) — True if any dep file was touched

    The range is exclusive of *good_sha* and inclusive of *bad_sha*, matching
    git bisect semantics (commits.json per schema 4.3).

    Backend selection:
      - If *repo_path* is provided (or neither backend is configured), use git.
      - If *github_repo* is provided and *repo_path* is None, use GitHub API.
    """
    if repo_path is not None:
        return _enumerate_commits_git(good_sha, bad_sha, repo_path=Path(repo_path))
    elif github_repo is not None:
        return _enumerate_commits_github(
            good_sha, bad_sha, github_repo=github_repo, github_token=github_token
        )
    else:
        raise ValueError(
            "enumerate_commits: at least one of repo_path or github_repo must be provided"
        )


def _enumerate_commits_git(
    good_sha: str,
    bad_sha: str,
    *,
    repo_path: Path,
) -> list[dict]:
    # git log newest-first; we reverse at the end.
    # Format: SHA|ISO-date|author-name|subject
    # The range good_sha..bad_sha gives commits AFTER good_sha up to and
    # including bad_sha (exclusive left, inclusive right).
    log_output = _git(
        [
            "log",
            "--format=%H|%ai|%an|%s",
            "--no-merges",
            f"{good_sha}..{bad_sha}",
        ],
        repo_path=repo_path,
    )

    commits = []
    for line in log_output.splitlines():
        line = line.strip()
        if not line:
            continue
        # Split on first 3 pipes only (subject may contain '|')
        parts = line.split("|", 3)
        if len(parts) < 4:
            # Fallback: treat as SHA only
            sha = parts[0].strip()
            date = ""
            author = ""
            message = ""
        else:
            sha, date, author, message = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()

        dep_changed = _git_dep_files_changed(sha, repo_path=repo_path)

        commits.append({
            "sha": sha,
            "short_sha": sha[:7],
            "date": date,
            "author": author,
            "message": message,
            "dep_files_changed": dep_changed,
        })

    # git log returns newest-first; reverse to oldest-first
    commits.reverse()
    return commits


def _enumerate_commits_github(
    good_sha: str,
    bad_sha: str,
    *,
    github_repo: str,
    github_token: str | None,
) -> list[dict]:
    """Use the GitHub Compare API to enumerate commits in range."""
    base_url = f"https://api.github.com/repos/{github_repo}"
    compare_url = f"{base_url}/compare/{good_sha}...{bad_sha}"
    data = _github_request(compare_url, token=github_token)

    raw_commits = data.get("commits", [])
    # GitHub compare returns oldest-first already.

    commits = []
    for c in raw_commits:
        sha = c["sha"]
        commit_info = c.get("commit", {})
        author_info = commit_info.get("author", {})
        date = author_info.get("date", "")
        author_name = author_info.get("name", "")
        message = commit_info.get("message", "").split("\n")[0]

        # Fetch full commit details to check files changed
        commit_detail = _github_request(
            f"{base_url}/commits/{sha}", token=github_token
        )
        dep_changed = _github_dep_files_changed(commit_detail)

        commits.append({
            "sha": sha,
            "short_sha": sha[:7],
            "date": date,
            "author": author_name,
            "message": message,
            "dep_files_changed": dep_changed,
        })

    return commits


# ---------------------------------------------------------------------------
# fetch_diff
# ---------------------------------------------------------------------------

def fetch_diff(
    sha_a: str,
    sha_b: str,
    *,
    repo_path: Path | None = None,
    github_repo: str | None = None,
    github_token: str | None = None,
) -> dict:
    """Return a diff summary between *sha_a* and *sha_b*.

    The result dict has:
        files_changed    (list[dict])  — [{path, additions, deletions}, ...]
        dep_files_changed (list[str])  — subset of changed paths that are dep files
        dep_changes      (list[str])   — +/- lines from dep file diffs (max 200)
        diff_summary     (str)         — full diff text, truncated at 8000 chars
        commit_message   (str)         — subject line of sha_b
    """
    if repo_path is not None:
        return _fetch_diff_git(sha_a, sha_b, repo_path=Path(repo_path))
    elif github_repo is not None:
        return _fetch_diff_github(
            sha_a, sha_b, github_repo=github_repo, github_token=github_token
        )
    else:
        raise ValueError(
            "fetch_diff: at least one of repo_path or github_repo must be provided"
        )


def _fetch_diff_git(sha_a: str, sha_b: str, *, repo_path: Path) -> dict:
    # ------ files_changed via git diff --stat ------
    stat_output = _git(
        ["diff", "--stat", "--no-color", f"{sha_a}..{sha_b}"],
        repo_path=repo_path,
    )

    files_changed: list[dict] = []
    for line in stat_output.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        # Format: " path/to/file.py | 12 +++---"
        path_part, stat_part = line.rsplit("|", 1)
        path = path_part.strip()
        stat_text = stat_part.strip()
        # Count + and - characters
        additions = stat_text.count("+")
        deletions = stat_text.count("-")
        files_changed.append({
            "path": path,
            "additions": additions,
            "deletions": deletions,
        })

    dep_files_changed = [f["path"] for f in files_changed if _is_dep_file(f["path"])]

    # ------ dep_changes: +/- lines from dep file diffs (max 200) ------
    dep_changes: list[str] = []
    if dep_files_changed:
        # Run git diff restricted to dep files
        dep_diff = _git(
            ["diff", "--no-color", f"{sha_a}..{sha_b}", "--"] + dep_files_changed,
            repo_path=repo_path,
        )
        for line in dep_diff.splitlines():
            if line.startswith("+") or line.startswith("-"):
                dep_changes.append(line)
            if len(dep_changes) >= 200:
                break

    # ------ diff_summary: full diff text truncated at 8000 chars ------
    full_diff = _git(
        ["diff", "--no-color", f"{sha_a}..{sha_b}"],
        repo_path=repo_path,
    )
    diff_summary = full_diff[:8000]

    # ------ commit_message: subject line of sha_b ------
    commit_message = _git(
        ["log", "--format=%s", "-1", sha_b],
        repo_path=repo_path,
    ).strip()

    return {
        "files_changed": files_changed,
        "dep_files_changed": dep_files_changed,
        "dep_changes": dep_changes,
        "diff_summary": diff_summary,
        "commit_message": commit_message,
    }


def _fetch_diff_github(
    sha_a: str,
    sha_b: str,
    *,
    github_repo: str,
    github_token: str | None,
) -> dict:
    base_url = f"https://api.github.com/repos/{github_repo}"
    compare_url = f"{base_url}/compare/{sha_a}...{sha_b}"
    data = _github_request(compare_url, token=github_token)

    raw_files = data.get("files", [])
    files_changed = [
        {
            "path": f.get("filename", ""),
            "additions": f.get("additions", 0),
            "deletions": f.get("deletions", 0),
        }
        for f in raw_files
    ]

    dep_files_changed = [f["path"] for f in files_changed if _is_dep_file(f["path"])]

    # dep_changes: collect +/- lines from patch fields of dep files
    dep_changes: list[str] = []
    for f in raw_files:
        if _is_dep_file(f.get("filename", "")):
            patch = f.get("patch", "")
            for line in patch.splitlines():
                if line.startswith("+") or line.startswith("-"):
                    dep_changes.append(line)
                if len(dep_changes) >= 200:
                    break
        if len(dep_changes) >= 200:
            break

    # diff_summary: concatenate all patches, truncated at 8000 chars
    parts = []
    total = 0
    for f in raw_files:
        header = f"--- {f.get('filename', '')}\n"
        patch = f.get("patch", "")
        chunk = header + patch + "\n"
        if total + len(chunk) > 8000:
            remaining = 8000 - total
            if remaining > len(header):
                parts.append(header + patch[: remaining - len(header)])
            break
        parts.append(chunk)
        total += len(chunk)
    diff_summary = "".join(parts)

    # commit_message: subject line of sha_b
    commit_data = _github_request(f"{base_url}/commits/{sha_b}", token=github_token)
    commit_message = commit_data.get("commit", {}).get("message", "").split("\n")[0]

    return {
        "files_changed": files_changed,
        "dep_files_changed": dep_files_changed,
        "dep_changes": dep_changes,
        "diff_summary": diff_summary,
        "commit_message": commit_message,
    }
