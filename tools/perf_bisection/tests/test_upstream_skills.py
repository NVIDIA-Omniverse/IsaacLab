# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for pinned upstream Agent Skill handoffs."""

from isaaclab_bisection.upstream_skills import (
    install_command,
    iter_skills,
    load_lock,
    main,
    skill_url,
    validate_lock,
)


def test_reviewed_upstream_skill_lock_is_valid() -> None:
    payload = load_lock()

    assert validate_lock(payload) == []
    assert payload["policy"]["execution_scope"] == "outer_agent_only"
    assert payload["policy"]["verdict_authority"] is False
    assert {skill["name"] for _, skill in iter_skills(payload)} == {
        "isaaclab-installing-isaac-lab",
        "isaaclab-setup-troubleshooting",
        "isaaclab-selecting-backends",
        "profile-isaac-sim",
    }


def test_install_command_uses_immutable_direct_skill_url() -> None:
    payload = load_lock()
    source, skill = next(iter_skills(payload))

    url = skill_url(source, skill)
    command = install_command(source, skill, "cursor", payload["installer"])

    assert source["commit_sha"] in url
    assert url.endswith(skill["path"])
    assert command == ["npx", "--yes", "skills@1.5.21", "add", url, "--agent", "cursor"]


def test_commands_can_be_limited_to_one_skill(capsys) -> None:
    exit_code = main(["commands", "--agent", "cursor", "--skill", "isaaclab-selecting-backends"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "isaaclab-selecting-backends" not in output
    assert "/skills/user/select-backends" in output
    assert "install-isaac-lab" not in output


def test_unknown_skill_is_rejected(capsys) -> None:
    exit_code = main(["commands", "--skill", "unknown"])

    assert exit_code == 2
    assert "unknown Skill name" in capsys.readouterr().out
