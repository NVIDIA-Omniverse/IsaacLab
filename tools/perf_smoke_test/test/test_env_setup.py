# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-free unit tests for the bisection environment-setup stack resolver.

These tests exercise the pin-extraction helpers on synthetic manifests and the
end-to-end :func:`resolve_stack` against real repository history, asserting that
it tolerates era drift (pins moving from ``setup.py`` to ``pyproject.toml``) and
discriminates stacks across eras. No GPU, network, or install is required.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

_GATE_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _GATE_DIR.parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

from bisection.env_setup import (  # noqa: E402
    _ARM_LIBGOMP_PATH,
    BENCHMARK_SUPPORT_LOCAL_PROJECTS,
    BENCHMARK_SUPPORT_PACKAGES,
    _benchmark_support_install_command,
    _benchmark_support_local_project_commands,
    _classify_install_failure,
    _python_requires_from_dir,
    _resolve_python_version,
    _spec_from_dir,
    _spec_from_manifest,
    _spec_from_setup_py,
    _split_requirement,
    resolve_stack,
    with_arm_libgomp_preload,
    with_uv_download_tuning,
)

# Commits with known pinned stacks, kept stable across eras.
_PR5265_GOOD = "1c237bdb"
_PR5265_BAD = "d1cb8e887"
_DEXSUITE_GOOD = "a9b62101"
_DEXSUITE_BAD = "8b63997a"


class TestSplitRequirement:
    """Name/specifier splitting for PEP 508-ish requirement strings."""

    @pytest.mark.parametrize(
        "requirement, name, specifier",
        [
            ("warp-lang==1.13.0", "warp-lang", "==1.13.0"),
            ("newton[sim]==1.2.1", "newton", "==1.2.1"),
            ("ovrtx>=0.3.0,<0.4.0", "ovrtx", ">=0.3.0,<0.4.0"),
            ("isaacsim[all,extscache]>=6.0.0", "isaacsim", ">=6.0.0"),
            ("ovphysx", "ovphysx", ""),
            ("torch>=2.0; python_version >= '3.10'", "torch", ">=2.0"),
            (
                "newton[sim] @ git+https://github.com/newton-physics/newton.git@v1.2.0",
                "newton",
                "@ git+https://github.com/newton-physics/newton.git@v1.2.0",
            ),
        ],
    )
    def test_split_extracts_name_and_specifier(self, requirement: str, name: str, specifier: str) -> None:
        assert _split_requirement(requirement) == (name, specifier)


class TestSpecFromManifest:
    """Version extraction from a declarative ``pyproject.toml``."""

    _PYPROJECT = """
        [project]
        name = "isaaclab"
        dependencies = ["warp-lang==1.13.0", "numpy"]
        [project.optional-dependencies]
        all = ["isaacsim[all]>=6.0.0"]
    """

    def test_reads_core_dependency(self) -> None:
        assert _spec_from_manifest(self._PYPROJECT, "warp-lang") == "==1.13.0"

    def test_reads_optional_dependency(self) -> None:
        assert _spec_from_manifest(self._PYPROJECT, "isaacsim") == ">=6.0.0"

    def test_missing_package_returns_none(self) -> None:
        assert _spec_from_manifest(self._PYPROJECT, "ovrtx") is None

    def test_invalid_toml_returns_none(self) -> None:
        assert _spec_from_manifest("this is = not [ valid toml", "warp-lang") is None

    def test_empty_input_returns_none(self) -> None:
        assert _spec_from_manifest(None, "warp-lang") is None


class TestSpecFromSetupPy:
    """Version extraction from an imperative ``setup.py`` via AST walking."""

    _SETUP = textwrap.dedent(
        '''
        """Installation script for the 'isaaclab_newton' python package. Don't edit."""
        INSTALL_REQUIRES = ["warp-lang==1.13.0"]
        EXTRAS = {
            "all": ["newton[sim] @ git+https://github.com/newton-physics/newton.git@v1.2.0"],
        }
        KEYWORDS = ["robotics", "newton"]
        setup(name="isaaclab_newton", install_requires=INSTALL_REQUIRES)
        '''
    )

    def test_reads_pinned_version(self) -> None:
        assert _spec_from_setup_py(self._SETUP, "warp-lang") == "==1.13.0"

    def test_reads_git_url_pin_despite_bare_keyword(self) -> None:
        # A bare ``"newton"`` keyword must not mask the real git-URL requirement.
        assert _spec_from_setup_py(self._SETUP, "newton") == "@ git+https://github.com/newton-physics/newton.git@v1.2.0"

    def test_syntax_error_returns_none(self) -> None:
        assert _spec_from_setup_py("def (:::", "warp-lang") is None

    def test_empty_input_returns_none(self) -> None:
        assert _spec_from_setup_py(None, "warp-lang") is None


class TestSpecFromDir:
    """``pyproject.toml`` takes precedence over ``setup.py`` when both pin a package."""

    def test_resolves_setup_py_pin_at_old_commit(self) -> None:
        # At the Dexsuite-era commit, newton lives only in setup.py as a git URL.
        spec = _spec_from_dir(_REPO_ROOT, _DEXSUITE_GOOD, "source/isaaclab_newton", "newton")
        assert spec == "@ git+https://github.com/newton-physics/newton.git@v1.2.0"

    def test_resolves_pyproject_pin_at_head(self) -> None:
        assert _spec_from_dir(_REPO_ROOT, "HEAD", "source/isaaclab_newton", "newton") == "==1.2.1"


class TestPythonRequires:
    """Resolution of the venv interpreter from a commit's ``requires-python``."""

    @pytest.mark.parametrize(
        "requires_python, expected",
        [
            (">=3.12", "3.12"),
            (">=3.10,<3.13", "3.10"),
            ("==3.11.*", "3.11"),
        ],
    )
    def test_resolves_lower_bound(self, requires_python: str, expected: str) -> None:
        assert _resolve_python_version(requires_python) == expected

    def test_unspecified_falls_back_to_host(self) -> None:
        assert _resolve_python_version(None) == f"{sys.version_info.major}.{sys.version_info.minor}"

    def test_reads_setup_py_python_requires_at_old_commit(self) -> None:
        # The Dexsuite-era commit declares ``python_requires=">=3.12"`` in setup.py.
        assert _python_requires_from_dir(_REPO_ROOT, _DEXSUITE_GOOD, "source/isaaclab") == ">=3.12"

    def test_reads_pyproject_requires_python_at_head(self) -> None:
        assert _python_requires_from_dir(_REPO_ROOT, "HEAD", "source/isaaclab") == ">=3.12"


class TestResolveStack:
    """End-to-end stack resolution against real repository history."""

    def test_isaacsim_resolves_from_env_base(self) -> None:
        stack = resolve_stack(_REPO_ROOT, _PR5265_GOOD)
        assert stack.isaacsim == "6.0.0-dev2"

    def test_resolves_setup_py_era_pins(self) -> None:
        stack = resolve_stack(_REPO_ROOT, _DEXSUITE_GOOD)
        assert stack.warp_lang == "==1.13.0"
        assert stack.newton == "@ git+https://github.com/newton-physics/newton.git@v1.2.0"
        assert stack.ovrtx == ">=0.3.0,<0.4.0"

    def test_resolves_pyproject_era_pins(self) -> None:
        stack = resolve_stack(_REPO_ROOT, "HEAD")
        assert stack.newton == "==1.2.1"
        assert stack.ovphysx == "==0.4.13"

    def test_resolves_python_from_commit_not_host(self) -> None:
        # IsaacLab requires 3.12 across these commits, read from the commit itself.
        stack = resolve_stack(_REPO_ROOT, _DEXSUITE_GOOD)
        assert stack.python_requires == ">=3.12"
        assert stack.python_version == "3.12"

    def test_hash_is_deterministic(self) -> None:
        assert resolve_stack(_REPO_ROOT, _PR5265_GOOD).stack_hash == resolve_stack(_REPO_ROOT, _PR5265_GOOD).stack_hash

    def test_known_good_and_bad_share_stack_within_a_range(self) -> None:
        # The regression is in code, not the pinned stack: a range's endpoints match.
        assert resolve_stack(_REPO_ROOT, _PR5265_GOOD).stack_hash == resolve_stack(_REPO_ROOT, _PR5265_BAD).stack_hash
        dexsuite_good = resolve_stack(_REPO_ROOT, _DEXSUITE_GOOD).stack_hash
        dexsuite_bad = resolve_stack(_REPO_ROOT, _DEXSUITE_BAD).stack_hash
        assert dexsuite_good == dexsuite_bad

    def test_distinct_eras_hash_differently(self) -> None:
        pr5265 = resolve_stack(_REPO_ROOT, _PR5265_GOOD).stack_hash
        dexsuite = resolve_stack(_REPO_ROOT, _DEXSUITE_GOOD).stack_hash
        head = resolve_stack(_REPO_ROOT, "HEAD").stack_hash
        assert len({pr5265, dexsuite, head}) == 3

    def test_to_json_round_trips_fields(self) -> None:
        stack = resolve_stack(_REPO_ROOT, "HEAD")
        payload = stack.to_json()
        assert payload["isaacsim"] == stack.isaacsim
        assert payload["stack_hash"] == stack.stack_hash


class TestBenchmarkSupportPackages:
    """Benchmark-only packages are installed into reconstructed envs."""

    def test_install_command_targets_reconstructed_python(self, tmp_path: Path) -> None:
        python_path = tmp_path / "env" / "bin" / "python"
        command = _benchmark_support_install_command(python_path)
        assert command[:4] == ["uv", "pip", "install", "--python"]
        assert command[4] == str(python_path)
        assert "psutil" in BENCHMARK_SUPPORT_PACKAGES
        assert "tensorboard" in BENCHMARK_SUPPORT_PACKAGES
        assert "h5py" in BENCHMARK_SUPPORT_PACKAGES
        assert "hydra-core" in BENCHMARK_SUPPORT_PACKAGES

    def test_local_project_commands_install_tasks_when_present(self, tmp_path: Path) -> None:
        python_path = tmp_path / "env" / "bin" / "python"
        for rel_path in BENCHMARK_SUPPORT_LOCAL_PROJECTS:
            project_dir = tmp_path / rel_path
            project_dir.mkdir(parents=True)
            (project_dir / "pyproject.toml").write_text("[project]\nname = 'dummy'\n", encoding="utf-8")

        commands = _benchmark_support_local_project_commands(tmp_path, python_path)

        assert len(commands) == len(BENCHMARK_SUPPORT_LOCAL_PROJECTS)
        assert all(command[:5] == ["uv", "pip", "install", "--python", str(python_path)] for command in commands)
        assert all("--editable" in command for command in commands)
        assert any(command[-1].endswith("source/isaaclab_tasks") for command in commands)

    def test_local_project_commands_skip_absent_projects(self, tmp_path: Path) -> None:
        python_path = tmp_path / "env" / "bin" / "python"
        assert _benchmark_support_local_project_commands(tmp_path, python_path) == []


class TestInstallFailureClassification:
    """Install failures distinguish retryable build friction from unavailable deps."""

    def test_arm_m64_compiler_failure_is_dependency_unavailable(self) -> None:
        log = "c++: error: unrecognized command-line option ‘-m64’"
        assert _classify_install_failure(log) == "dependency_unavailable"

    def test_arm_libgomp_preload_is_appended_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_libgomp = tmp_path / "libgomp.so.1"
        fake_libgomp.write_text("", encoding="utf-8")
        monkeypatch.setattr("bisection.env_setup._ARM_LIBGOMP_PATH", fake_libgomp)
        monkeypatch.setattr("platform.machine", lambda: "aarch64")

        env = with_arm_libgomp_preload({"LD_PRELOAD": "/already/preloaded.so"})
        assert env["LD_PRELOAD"] == f"/already/preloaded.so:{fake_libgomp}"
        assert with_arm_libgomp_preload(env)["LD_PRELOAD"] == env["LD_PRELOAD"]

    def test_arm_libgomp_preload_skips_non_arm(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("bisection.env_setup._ARM_LIBGOMP_PATH", _ARM_LIBGOMP_PATH)
        monkeypatch.setattr("platform.machine", lambda: "x86_64")
        assert with_arm_libgomp_preload({"A": "B"}) == {"A": "B"}

    def test_uv_download_tuning_pins_shared_cache_and_timeout(self, tmp_path: Path) -> None:
        tuned = with_uv_download_tuning({"PATH": "/usr/bin"}, cache_root=tmp_path)
        assert tuned["PATH"] == "/usr/bin"
        assert tuned["UV_CACHE_DIR"] == str(tmp_path / "uv-cache")
        assert tuned["UV_PYTHON_INSTALL_DIR"] == str(tmp_path / "uv-python")
        assert int(tuned["UV_HTTP_TIMEOUT"]) >= 300

    def test_uv_download_tuning_respects_caller_overrides(self, tmp_path: Path) -> None:
        env = {
            "UV_CACHE_DIR": "/custom/cache",
            "UV_PYTHON_INSTALL_DIR": "/custom/python",
            "UV_HTTP_TIMEOUT": "42",
        }
        tuned = with_uv_download_tuning(env, cache_root=tmp_path)
        assert tuned["UV_CACHE_DIR"] == "/custom/cache"
        assert tuned["UV_PYTHON_INSTALL_DIR"] == "/custom/python"
        assert tuned["UV_HTTP_TIMEOUT"] == "42"
