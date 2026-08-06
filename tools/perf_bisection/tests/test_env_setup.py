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

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_GATE_DIR = Path(__file__).resolve().parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

from isaaclab_bisection.bisection.env_setup import (  # noqa: E402
    _ARM_LIBGOMP_PATH,
    BENCHMARK_SUPPORT_LOCAL_PROJECTS,
    BENCHMARK_SUPPORT_PACKAGES,
    _benchmark_support_install_command,
    _benchmark_support_local_project_commands,
    _candidate_install_command,
    _classify_install_failure,
    _import_verification_code,
    _legacy_isaacsim_install_command,
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


@pytest.fixture
def stack_repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Create self-contained setup.py and pyproject stack eras."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)

    def write(relative_path: str, content: str) -> None:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content), encoding="utf-8")

    def commit(message: str) -> str:
        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(["git", "commit", "-m", message], cwd=root, check=True, capture_output=True)
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()

    write("docker/.env.base", "ISAACSIM_VERSION=6.0.0-dev2\n")
    write(
        "source/isaaclab/setup.py",
        """
        setup(
            name="isaaclab",
            python_requires=">=3.12",
            install_requires=["warp-lang==1.13.0"],
        )
        """,
    )
    write(
        "source/isaaclab_newton/setup.py",
        """
        setup(
            name="isaaclab_newton",
            install_requires=["newton[sim] @ git+https://github.com/newton-physics/newton.git@v1.2.0"],
        )
        """,
    )
    write(
        "source/isaaclab_ov/setup.py",
        """
        setup(name="isaaclab_ov", install_requires=["ovrtx>=0.3.0,<0.4.0"])
        """,
    )
    setup_good = commit("Add setup.py stack")
    write("marker.txt", "same stack, different source\n")
    setup_bad = commit("Change source only")

    shutil.rmtree(root / "source")
    write("docker/.env.base", "ISAACSIM_VERSION=6.0.0\n")
    write(
        "source/isaaclab/pyproject.toml",
        """
        [project]
        name = "isaaclab"
        version = "3.0.0"
        requires-python = ">=3.12"
        dependencies = ["warp-lang==1.14.0"]
        """,
    )
    write(
        "source/isaaclab_newton/pyproject.toml",
        """
        [project]
        name = "isaaclab-newton"
        version = "3.0.0"
        dependencies = ["newton[sim]==1.2.1"]
        """,
    )
    write(
        "source/isaaclab_ov/pyproject.toml",
        """
        [project]
        name = "isaaclab-ov"
        version = "3.0.0"
        dependencies = ["ovrtx==0.4.0"]
        """,
    )
    write(
        "source/isaaclab_ovphysx/pyproject.toml",
        """
        [project]
        name = "isaaclab-ovphysx"
        version = "3.0.0"
        dependencies = ["ovphysx==0.4.13"]
        """,
    )
    pyproject = commit("Migrate stack to pyproject")
    return root, {"setup_good": setup_good, "setup_bad": setup_bad, "pyproject": pyproject}


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

    def test_resolves_setup_py_pin_at_old_commit(self, stack_repo: tuple[Path, dict[str, str]]) -> None:
        root, commits = stack_repo
        spec = _spec_from_dir(root, commits["setup_good"], "source/isaaclab_newton", "newton")
        assert spec == "@ git+https://github.com/newton-physics/newton.git@v1.2.0"

    def test_resolves_pyproject_pin(self, stack_repo: tuple[Path, dict[str, str]]) -> None:
        root, commits = stack_repo
        assert _spec_from_dir(root, commits["pyproject"], "source/isaaclab_newton", "newton") == "==1.2.1"


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

    @pytest.mark.parametrize(
        "isaacsim_version, expected",
        [
            ("4.5.0", "3.10"),
            ("5.1.0", "3.11"),
            ("6.0.0-dev2", "3.12"),
        ],
    )
    def test_isaacsim_pin_selects_required_python_abi(self, isaacsim_version: str, expected: str) -> None:
        assert _resolve_python_version(">=3.10", isaacsim_version) == expected

    def test_reads_setup_py_python_requires_at_old_commit(self, stack_repo: tuple[Path, dict[str, str]]) -> None:
        root, commits = stack_repo
        assert _python_requires_from_dir(root, commits["setup_good"], "source/isaaclab") == ">=3.12"

    def test_reads_pyproject_requires_python(self, stack_repo: tuple[Path, dict[str, str]]) -> None:
        root, commits = stack_repo
        assert _python_requires_from_dir(root, commits["pyproject"], "source/isaaclab") == ">=3.12"


class TestResolveStack:
    """End-to-end stack resolution against self-contained repository history."""

    def test_isaacsim_resolves_from_env_base(self, stack_repo: tuple[Path, dict[str, str]]) -> None:
        root, commits = stack_repo
        stack = resolve_stack(root, commits["setup_good"])
        assert stack.isaacsim == "6.0.0-dev2"

    def test_resolves_setup_py_era_pins(self, stack_repo: tuple[Path, dict[str, str]]) -> None:
        root, commits = stack_repo
        stack = resolve_stack(root, commits["setup_good"])
        assert stack.warp_lang == "==1.13.0"
        assert stack.newton == "@ git+https://github.com/newton-physics/newton.git@v1.2.0"
        assert stack.ovrtx == ">=0.3.0,<0.4.0"

    def test_resolves_pyproject_era_pins(self, stack_repo: tuple[Path, dict[str, str]]) -> None:
        root, commits = stack_repo
        stack = resolve_stack(root, commits["pyproject"])
        assert stack.newton == "==1.2.1"
        assert stack.ovphysx == "==0.4.13"

    def test_resolves_python_from_commit_not_host(self, stack_repo: tuple[Path, dict[str, str]]) -> None:
        root, commits = stack_repo
        stack = resolve_stack(root, commits["setup_good"])
        assert stack.python_requires == ">=3.12"
        assert stack.python_version == "3.12"

    def test_hash_is_deterministic(self, stack_repo: tuple[Path, dict[str, str]]) -> None:
        root, commits = stack_repo
        assert (
            resolve_stack(root, commits["setup_good"]).stack_hash
            == resolve_stack(root, commits["setup_good"]).stack_hash
        )

    def test_known_good_and_bad_share_stack_within_a_range(self, stack_repo: tuple[Path, dict[str, str]]) -> None:
        root, commits = stack_repo
        assert (
            resolve_stack(root, commits["setup_good"]).stack_hash
            == resolve_stack(root, commits["setup_bad"]).stack_hash
        )

    def test_distinct_eras_hash_differently(self, stack_repo: tuple[Path, dict[str, str]]) -> None:
        root, commits = stack_repo
        setup = resolve_stack(root, commits["setup_good"]).stack_hash
        pyproject = resolve_stack(root, commits["pyproject"]).stack_hash
        assert setup != pyproject

    def test_to_json_round_trips_fields(self, stack_repo: tuple[Path, dict[str, str]]) -> None:
        root, commits = stack_repo
        stack = resolve_stack(root, commits["pyproject"])
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

    def test_candidate_install_command_uses_none_for_legacy_rl_extra_contract(self, tmp_path: Path) -> None:
        script = tmp_path / "isaaclab.sh"
        script.write_text(
            'framework_name=$2\n${pip_command} -e "${ISAACLAB_PATH}/source/isaaclab_rl[${framework_name}]"\n',
            encoding="utf-8",
        )

        command = _candidate_install_command(tmp_path, "newton,ov[ovrtx],isaacsim")

        assert command == [str(script), "-i", "none"]

    def test_candidate_install_command_preserves_component_scope_for_current_contract(self, tmp_path: Path) -> None:
        script = tmp_path / "isaaclab.sh"
        script.write_text('install_isaaclab --include "$2"\n', encoding="utf-8")

        command = _candidate_install_command(tmp_path, "newton,ov[ovrtx],isaacsim")

        assert command == [str(script), "-i", "newton,ov[ovrtx],isaacsim"]

    def test_legacy_isaacsim_install_command_pins_version_and_nvidia_index(self, tmp_path: Path) -> None:
        command = _legacy_isaacsim_install_command(tmp_path / "bin" / "python", "5.1.0")

        assert "isaacsim[all,extscache]==5.1.0" in command
        assert "https://pypi.nvidia.com" in command
        assert "unsafe-best-match" in command

    def test_legacy_import_verification_starts_simulation_app_before_tasks(self, tmp_path: Path) -> None:
        (tmp_path / "isaaclab.sh").write_text(
            'framework_name=$2\n${pip_command} -e "${ISAACLAB_PATH}/source/isaaclab_rl[${framework_name}]"\n',
            encoding="utf-8",
        )

        code = _import_verification_code(tmp_path)

        assert "SimulationApp({'headless': True})" in code
        assert code.index("SimulationApp") < code.index("import isaaclab_tasks")
        assert code.endswith("simulation_app.close()")

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
        monkeypatch.setattr("isaaclab_bisection.bisection.env_setup._ARM_LIBGOMP_PATH", fake_libgomp)
        monkeypatch.setattr("platform.machine", lambda: "aarch64")

        env = with_arm_libgomp_preload({"LD_PRELOAD": "/already/preloaded.so"})
        assert env["LD_PRELOAD"] == f"/already/preloaded.so:{fake_libgomp}"
        assert with_arm_libgomp_preload(env)["LD_PRELOAD"] == env["LD_PRELOAD"]

    def test_arm_libgomp_preload_skips_non_arm(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("isaaclab_bisection.bisection.env_setup._ARM_LIBGOMP_PATH", _ARM_LIBGOMP_PATH)
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
