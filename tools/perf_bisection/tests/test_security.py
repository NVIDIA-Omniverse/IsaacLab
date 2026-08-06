# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for security boundaries around candidate code and optional LLM use."""

from pathlib import Path

import pytest

from isaaclab_bisection.bisection.models import BisectionPlan, RunnerSpec
from isaaclab_bisection.bisection.security import (
    candidate_subprocess_environment,
    parse_probe_debug_command,
    redact_sensitive_text,
    resolve_path_within,
    validate_llm_base_url,
    validate_relative_path,
)


def test_candidate_environment_drops_credentials_and_unrelated_values() -> None:
    source = {
        "PATH": "/usr/bin",
        "LANG": "C.UTF-8",
        "OPENAI_API_KEY": "llm-secret",
        "GITHUB_TOKEN": "github-secret",
        "SSH_AUTH_SOCK": "/tmp/agent.sock",
        "UNRELATED_SETTING": "private",
    }

    assert candidate_subprocess_environment(source) == {"PATH": "/usr/bin", "LANG": "C.UTF-8"}


def test_candidate_environment_rejects_credentialed_proxy_url() -> None:
    source = {
        "HTTPS_PROXY": "https://user:password@proxy.example.com",
        "NO_PROXY": "localhost",
    }

    assert candidate_subprocess_environment(source) == {"NO_PROXY": "localhost"}


@pytest.mark.parametrize("command", ["df -h", "docker info", "docker version", "nvidia-smi"])
def test_probe_debug_command_accepts_read_only_allowlist(command: str) -> None:
    assert parse_probe_debug_command(command)


@pytest.mark.parametrize(
    "command",
    [
        "curl https://example.com",
        "df -h; env",
        "python -c 'print(1)'",
        "rm -rf /",
    ],
)
def test_probe_debug_command_rejects_arbitrary_shell(command: str) -> None:
    with pytest.raises(ValueError, match="not allowlisted"):
        parse_probe_debug_command(command)


def test_llm_endpoint_must_be_explicit_and_secure() -> None:
    with pytest.raises(ValueError, match="explicit --base_url"):
        validate_llm_base_url(None)
    with pytest.raises(ValueError, match="must use HTTPS"):
        validate_llm_base_url("http://models.example.com/v1")
    with pytest.raises(ValueError, match="must not contain credentials"):
        validate_llm_base_url("https://user:secret@models.example.com/v1")
    with pytest.raises(ValueError, match="private, link-local, or reserved"):
        validate_llm_base_url("https://169.254.169.254/v1")

    assert validate_llm_base_url("https://inference.nvidia.com/v1/") == "https://inference.nvidia.com/v1"
    assert validate_llm_base_url("http://localhost:8000/v1") == "http://localhost:8000/v1"


def test_llm_endpoint_honors_deployment_host_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISAACLAB_BISECTION_LLM_HOSTS", "inference.nvidia.com")

    assert validate_llm_base_url("https://inference.nvidia.com/v1") == "https://inference.nvidia.com/v1"
    with pytest.raises(ValueError, match="not in ISAACLAB_BISECTION_LLM_HOSTS"):
        validate_llm_base_url("https://models.example.com/v1")


def test_run_paths_cannot_escape_or_traverse(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()

    assert validate_relative_path("tooling/perf_smoke_test", "snapshot")
    with pytest.raises(ValueError, match="non-traversing"):
        validate_relative_path("../../outside", "snapshot")
    with pytest.raises(ValueError, match="below the run root"):
        resolve_path_within(root, tmp_path / "outside", "source_dir")
    (root / "linked").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="must not be a symbolic link"):
        resolve_path_within(root, "linked", "source_dir")


def test_plan_identifiers_and_runner_extra_args_are_constrained() -> None:
    with pytest.raises(ValueError, match="not allowlisted"):
        RunnerSpec(extra_args=["--repo_root", "/tmp/other"])
    with pytest.raises(ValueError, match="safe path component"):
        BisectionPlan(
            task_id="../../escape",
            backend_key="physx",
            good_ref="good",
            bad_ref="bad",
            gpu_model="gpu",
        )

    assert RunnerSpec(extra_args=["--install_scope", "newton,isaacsim"])


def test_sensitive_text_is_redacted_before_llm_handoff() -> None:
    text = "Authorization: Bearer top-secret\napi_key=abc123\ndownload=https://user:password@example.com/file\n"

    redacted = redact_sensitive_text(text)

    assert "top-secret" not in redacted
    assert "abc123" not in redacted
    assert "user:password" not in redacted
