# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate and print reviewed installation commands for pinned upstream Agent Skills."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_LOCK_PATH = Path(__file__).with_name("upstream_skills.lock.json")
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_HANDOFFS = {
    "host_onboarding",
    "host_setup_failure",
    "plan_backend_selection",
    "post_bisection_profiling",
}


def load_lock(path: Path = _LOCK_PATH) -> dict[str, Any]:
    """Load the pinned upstream Skill manifest."""
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def iter_skills(payload: dict[str, Any]) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield each ``(source, skill)`` pair in the manifest."""
    for source in payload.get("sources", []):
        for skill in source.get("skills", []):
            yield source, skill


def validate_lock(payload: dict[str, Any]) -> list[str]:
    """Return validation errors for an upstream Skill manifest."""
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    policy = payload.get("policy")
    if not isinstance(policy, dict):
        errors.append("policy must be an object")
    elif policy.get("verdict_authority") is not False:
        errors.append("upstream Skills must not have verdict authority")

    installer = payload.get("installer")
    if not isinstance(installer, dict):
        errors.append("installer must be an object")
    else:
        if installer.get("package") != "skills":
            errors.append("installer.package must be skills")
        version = installer.get("version")
        if not isinstance(version, str) or not _SEMVER.fullmatch(version):
            errors.append("installer.version must be an exact semantic version")
        integrity = installer.get("integrity")
        if not isinstance(integrity, str) or not integrity.startswith("sha512-"):
            errors.append("installer.integrity must be an sha512 npm integrity value")

    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        return [*errors, "sources must be a non-empty list"]

    names: set[str] = set()
    for source_index, source in enumerate(sources):
        prefix = f"sources[{source_index}]"
        if not isinstance(source, dict):
            errors.append(f"{prefix} must be an object")
            continue
        repository = source.get("repository")
        if not isinstance(repository, str) or not repository.startswith("https://github.com/"):
            errors.append(f"{prefix}.repository must be a public GitHub URL")
        commit_sha = source.get("commit_sha")
        if not isinstance(commit_sha, str) or not _FULL_SHA.fullmatch(commit_sha):
            errors.append(f"{prefix}.commit_sha must be a full lowercase commit SHA")
        skills = source.get("skills")
        if not isinstance(skills, list) or not skills:
            errors.append(f"{prefix}.skills must be a non-empty list")
            continue
        for skill_index, skill in enumerate(skills):
            skill_prefix = f"{prefix}.skills[{skill_index}]"
            if not isinstance(skill, dict):
                errors.append(f"{skill_prefix} must be an object")
                continue
            name = skill.get("name")
            if not isinstance(name, str) or not name:
                errors.append(f"{skill_prefix}.name must be a non-empty string")
            elif name in names:
                errors.append(f"duplicate Skill name: {name}")
            else:
                names.add(name)
            path = skill.get("path")
            if not isinstance(path, str) or not path.startswith("skills/") or ".." in Path(path).parts:
                errors.append(f"{skill_prefix}.path must stay below skills/")
            if skill.get("handoff") not in _HANDOFFS:
                errors.append(f"{skill_prefix}.handoff is not recognized")
            if not isinstance(skill.get("restriction"), str) or not skill["restriction"]:
                errors.append(f"{skill_prefix}.restriction must be a non-empty string")
    return errors


def skill_url(source: dict[str, Any], skill: dict[str, Any]) -> str:
    """Return the immutable GitHub URL for one pinned Skill."""
    return f"{source['repository'].rstrip('/')}/tree/{source['commit_sha']}/{skill['path']}"


def install_command(source: dict[str, Any], skill: dict[str, Any], agent: str, installer: dict[str, str]) -> list[str]:
    """Build a version-pinned Skills CLI command that retains human confirmation."""
    package = f"{installer['package']}@{installer['version']}"
    return ["npx", "--yes", package, "add", skill_url(source, skill), "--agent", agent]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=_LOCK_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="Validate the lock manifest.")
    subparsers.add_parser("list", help="List pinned Skill names and handoffs.")
    commands = subparsers.add_parser("commands", help="Print reviewed, version-pinned Skills CLI commands.")
    commands.add_argument("--agent", default="cursor", help="Skills CLI agent identifier.")
    commands.add_argument("--skill", action="append", default=[], help="Limit output to one or more Skill names.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the upstream Skill manifest utility."""
    args = _parse_args(argv)
    payload = load_lock(args.lock)
    errors = validate_lock(payload)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    entries = list(iter_skills(payload))
    if args.command == "validate":
        print(f"Validated {len(entries)} pinned upstream Skills.")
        return 0
    if args.command == "list":
        for _, skill in entries:
            print(f"{skill['name']}\t{skill['handoff']}\t{skill['restriction']}")
        return 0

    selected = set(args.skill)
    known = {skill["name"] for _, skill in entries}
    unknown = selected - known
    if unknown:
        print(f"ERROR: unknown Skill name(s): {', '.join(sorted(unknown))}")
        return 2
    for source, skill in entries:
        if not selected or skill["name"] in selected:
            print(shlex.join(install_command(source, skill, args.agent, payload["installer"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
