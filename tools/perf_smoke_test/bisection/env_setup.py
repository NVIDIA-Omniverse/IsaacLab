# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-commit environment reconstruction for the IsaacLab bisection agent.

The bisection agent must measure each candidate commit against the *exact*
runtime environment that commit pinned, not whatever happens to be installed on
the host. Reusing the host environment is unfaithful in two ways:

* every IsaacLab submodule except core ``isaaclab`` is editable-installed against
  the host working tree, so the host's code (not the checked-out commit's) is what
  runs; and
* the host's Isaac Sim is a single fixed version, which silently varies a second
  dimension when a commit pinned a different Isaac Sim.

This module therefore reconstructs a fully isolated environment per commit:

* :func:`resolve_stack` reads the versions a commit pinned (Isaac Sim plus the
  modular ``warp``/``newton``/``ovrtx``/``ovphysx`` layer) straight from that
  commit's tree via ``git show`` -- no checkout required -- and is tolerant of the
  pin locations drifting across eras.
* :func:`ensure_env` builds a dedicated ``uv`` virtual environment and installs the
  commit's own clone via its ``./isaaclab.sh -i`` so the pinned stack *and* the
  commit's code are reproduced. Builds are per-commit (the editable installs bind
  to the clone) but the multi-GB downloads are amortized across commits by uv's
  global cache and hardlinking.

When a commit pins something that can no longer be installed, the build raises
:class:`EnvSkip` with a category so the engine can skip the commit gracefully
rather than misclassifying it as a regression.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import tomllib

from .git_utils import git, resolve_ref

try:  # parent-level helpers; perf_smoke_test is on sys.path when bisection is imported
    from hashing import stable_hash
    from image_era import parse_env_file, read_env_base_from_commit
except ImportError:  # pragma: no cover - fallback for namespace-package import
    from ..hashing import stable_hash
    from ..image_era import parse_env_file, read_env_base_from_commit

# Default ``./isaaclab.sh -i`` scope. The core submodules (incl. isaaclab_tasks and
# the renderer packages) are editable-installed on every ``-i``; the tokens here
# add the heavy third-party layer that ``-i all`` would otherwise omit:
#   - ``newton``        -> newton[sim] physics backend
#   - ``ov[ovrtx]``     -> ovrtx + ovphysx renderer bindings (``-i all`` excludes ``ov``)
#   - ``isaacsim``      -> the pinned Isaac Sim itself (``install_isaacsim`` defaults off)
DEFAULT_INSTALL_SCOPE = "newton,ov[ovrtx],isaacsim"
_PROGRESS_PREFIX = "[perf-bisect]"

# The bisection runner executes ``tools/perf_smoke_test/perf_runtime.py``, which
# imports ``isaaclab.test.benchmark`` recorders. Some historical commits did not
# declare every recorder dependency in install metadata; install the small support
# packages here so environment friction does not become a false regression signal.
BENCHMARK_SUPPORT_PACKAGES = ("psutil", "tensorboard", "h5py", "hydra-core")
BENCHMARK_SUPPORT_LOCAL_PROJECTS = (
    "source/isaaclab_assets",
    "source/isaaclab_contrib",
    "source/isaaclab_ovphysx",
    "source/isaaclab_physx",
    "source/isaaclab_tasks",
)

# Submodule directories consulted for each modular pin. Each is read tolerant of
# era drift: modern commits declare pins in ``pyproject.toml``, older ones in
# ``setup.py`` (often as a ``git+...@<ref>`` URL), so both are tried.
_CORE_DIR = "source/isaaclab"
_NEWTON_DIRS = ("source/isaaclab_newton", "source/isaaclab_physx")
_OV_DIR = "source/isaaclab_ov"
_OVPHYSX_DIR = "source/isaaclab_ovphysx"


def _progress(message: str) -> None:
    """Emit a structured environment-setup milestone in verbose mode."""
    if os.environ.get("PERF_BISECT_PROGRESS") == "verbose":
        print(f"{_PROGRESS_PREFIX} {message}", flush=True)


# Install-log signatures that distinguish an unavailable pin from a build failure.
_UNAVAILABLE_MARKERS = (
    "no solution found",
    "no matching distribution",
    "could not find a version",
    "not found in the package registry",
    "does not exist",
)
_UNSUPPORTED_PLATFORM_MARKERS = (
    # Some source distributions are not portable to the local architecture. For
    # example, pytetwild/geogram emits x86-only ``-m64`` on aarch64.
    "unrecognized command-line option ‘-m64’",
    "unrecognized command-line option '-m64'",
)
_ARM_LIBGOMP_PATH = Path("/lib/aarch64-linux-gnu/libgomp.so.1")


@dataclass(frozen=True)
class StackSpec:
    """The runtime versions a commit pinned, plus a content hash for reporting.

    Version fields hold the requirement specifier as written (e.g. ``"==1.13.0"``,
    ``">=6.0.0"``) or ``None`` when the pin could not be located at that commit.
    ``python_requires`` is the commit's raw ``requires-python`` / ``python_requires``
    specifier; ``python_version`` is the concrete ``X.Y`` resolved from it (the venv
    interpreter), falling back to the host interpreter when unspecified.
    """

    commit_sha: str
    isaacsim: str | None
    warp_lang: str | None
    newton: str | None
    ovrtx: str | None
    ovphysx: str | None
    python_requires: str | None
    python_version: str
    platform: str
    stack_hash: str

    def to_json(self) -> dict:
        """Serialize the stack spec to JSON."""
        return asdict(self)


@dataclass(frozen=True)
class EnvHandle:
    """A reconstructed environment ready to run a benchmark."""

    python_path: str
    env_dir: str
    stack_hash: str
    isaacsim_version: str | None
    reused: bool

    def to_json(self) -> dict:
        """Serialize the env handle to JSON."""
        return asdict(self)


class EnvSkip(Exception):
    """Raised when a commit's environment cannot be reconstructed.

    Args:
        category: One of ``"dependency_unavailable"``, ``"install_failed"``, or
            ``"runtime_incompatible"`` -- the bisection engine treats all three as
            a skip (never a regression).
        detail: Human-readable cause (e.g. the offending pin and the failing line).
    """

    def __init__(self, category: str, detail: str) -> None:
        super().__init__(f"{category}: {detail}")
        self.category = category
        self.detail = detail


def _git_show(repo_root: Path, commit: str, rel_path: str) -> str | None:
    """Return file contents at a commit via ``git show``, or ``None`` if absent."""
    result = git(repo_root, ["show", f"{commit}:{rel_path}"], check=False)
    return result.stdout if result.returncode == 0 else None


def _split_requirement(requirement: str) -> tuple[str, str]:
    """Split a PEP 508 requirement into ``(normalized_name, version_specifier)``.

    Extras and environment markers are stripped; the specifier is whatever remains
    after the distribution name (e.g. ``"newton[sim]==1.2.1"`` -> ``("newton",
    "==1.2.1")``).
    """
    text = requirement.split(";", 1)[0].strip()
    name_chars: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in "._-":
            name_chars.append(ch)
        else:
            break
    name = "".join(name_chars)
    specifier = text[len(name) :].strip()
    if specifier.startswith("["):  # drop extras, keep the version specifier after ']'
        specifier = specifier[specifier.index("]") + 1 :].strip() if "]" in specifier else ""
    normalized = name.lower().replace("_", "-")
    return normalized, specifier


def _spec_from_manifest(text: str | None, package: str) -> str | None:
    """Return the version specifier pinned for ``package`` in a pyproject's deps.

    Scans both ``[project.dependencies]`` and every ``[project.optional-dependencies]``
    group. Returns ``None`` when the package is absent or the TOML cannot be parsed.
    """
    if not text:
        return None
    try:
        data = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, ValueError):
        return None
    project = data.get("project", {})
    requirements: list[str] = list(project.get("dependencies", []) or [])
    for group in (project.get("optional-dependencies", {}) or {}).values():
        requirements.extend(group or [])
    target = package.lower().replace("_", "-")
    for requirement in requirements:
        if not isinstance(requirement, str):
            continue
        name, specifier = _split_requirement(requirement)
        if name == target:
            return specifier or None
    return None


def _spec_from_setup_py(text: str | None, package: str) -> str | None:
    """Return the version specifier pinned for ``package`` in a ``setup.py``.

    Parses the file with :mod:`ast` and walks its string literals (``setup.py`` is
    code, not declarative TOML, so quote-matching is unreliable). Returns the first
    non-empty specifier for the package -- so a bare keyword like ``"newton"`` does
    not mask a real ``"newton[sim] @ git+...@v1.2.0"`` pin.
    """
    if not text:
        return None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    target = package.lower().replace("_", "-")
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            name, specifier = _split_requirement(node.value)
            if name == target and specifier:
                return specifier
    return None


def _spec_from_dir(repo_root: Path, commit: str, dir_rel: str, package: str) -> str | None:
    """Resolve a pin from a submodule dir, trying ``pyproject.toml`` then ``setup.py``."""
    spec = _spec_from_manifest(_git_show(repo_root, commit, f"{dir_rel}/pyproject.toml"), package)
    if spec:
        return spec
    return _spec_from_setup_py(_git_show(repo_root, commit, f"{dir_rel}/setup.py"), package)


# Matches the first ``major.minor`` in a ``requires-python`` specifier (its lower bound).
_PY_VERSION_RE = re.compile(r"(\d+)\.(\d+)")


def _python_requires_from_dir(repo_root: Path, commit: str, dir_rel: str) -> str | None:
    """Return a commit's ``requires-python`` specifier, era-tolerant.

    Reads ``[project].requires-python`` from ``pyproject.toml``; falls back to the
    ``python_requires=`` keyword of the ``setup()`` call in ``setup.py``.
    """
    manifest = _git_show(repo_root, commit, f"{dir_rel}/pyproject.toml")
    if manifest:
        try:
            spec = tomllib.loads(manifest).get("project", {}).get("requires-python")
        except (tomllib.TOMLDecodeError, ValueError):
            spec = None
        if spec:
            return spec
    setup_py = _git_show(repo_root, commit, f"{dir_rel}/setup.py")
    if setup_py:
        try:
            tree = ast.parse(setup_py)
        except SyntaxError:
            return None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.keyword)
                and node.arg == "python_requires"
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                return node.value.value
    return None


def _resolve_python_version(requires_python: str | None) -> str:
    """Resolve a concrete ``X.Y`` interpreter from a specifier, defaulting to the host.

    Picks the lower bound (e.g. ``">=3.12"`` -> ``"3.12"``, ``">=3.10,<3.13"`` ->
    ``"3.10"``) so the venv uses the oldest interpreter the commit still supports.
    """
    if requires_python:
        match = _PY_VERSION_RE.search(requires_python)
        if match:
            return f"{match.group(1)}.{match.group(2)}"
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def resolve_stack(repo_root: Path | str, commit_sha: str) -> StackSpec:
    """Resolve the runtime versions a commit pinned, tolerant of era drift.

    Isaac Sim is read from ``docker/.env.base`` (the most stable, era-spanning
    source) and falls back to the ``isaacsim`` extra in the core pyproject. The
    modular pins are read from their respective manifests. Missing pins degrade to
    ``None`` rather than failing, since the actual install is driven by the clone's
    own manifests; the resolved versions feed the content hash and reporting.

    Args:
        repo_root: Repository root used for ``git show`` lookups.
        commit_sha: Commit (any git ref) whose pinned stack to resolve.

    Returns:
        The resolved :class:`StackSpec` including a deterministic ``stack_hash``.
    """
    root = Path(repo_root)
    commit = resolve_ref(root, commit_sha)

    isaacsim: str | None = None
    try:
        isaacsim = parse_env_file(read_env_base_from_commit(commit, root)).get("ISAACSIM_VERSION") or None
    except FileNotFoundError:
        isaacsim = None

    if isaacsim is None:
        isaacsim = _spec_from_dir(root, commit, _CORE_DIR, "isaacsim")
    warp_lang = _spec_from_dir(root, commit, _CORE_DIR, "warp-lang")

    newton: str | None = None
    for newton_dir in _NEWTON_DIRS:
        newton = _spec_from_dir(root, commit, newton_dir, "newton")
        if newton:
            break

    ovrtx = _spec_from_dir(root, commit, _OV_DIR, "ovrtx")
    ovphysx = _spec_from_dir(root, commit, _OVPHYSX_DIR, "ovphysx")

    python_requires = _python_requires_from_dir(root, commit, _CORE_DIR)
    python_version = _resolve_python_version(python_requires)
    machine = platform.machine()
    version_map = {
        "isaacsim": isaacsim,
        "warp_lang": warp_lang,
        "newton": newton,
        "ovrtx": ovrtx,
        "ovphysx": ovphysx,
        "python_version": python_version,
        "platform": machine,
    }
    return StackSpec(
        commit_sha=commit,
        isaacsim=isaacsim,
        warp_lang=warp_lang,
        newton=newton,
        ovrtx=ovrtx,
        ovphysx=ovphysx,
        python_requires=python_requires,
        python_version=python_version,
        platform=machine,
        stack_hash=stable_hash(version_map),
    )


def _classify_install_failure(log_text: str) -> str:
    """Classify an install failure as unavailable-dependency vs. build failure."""
    lowered = log_text.lower()
    if any(marker in lowered for marker in (*_UNAVAILABLE_MARKERS, *_UNSUPPORTED_PLATFORM_MARKERS)):
        return "dependency_unavailable"
    return "install_failed"


def _tail(text: str, *, max_lines: int = 8) -> str:
    """Return the last non-empty lines of ``text`` for a concise skip detail."""
    lines = [line for line in text.splitlines() if line.strip()]
    return " | ".join(lines[-max_lines:]) if lines else ""


def _run_logged(
    command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path, timeout_s: int
) -> tuple[int, str]:
    """Run a command, tee combined output to ``log_path``, and return ``(rc, output)``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_fh:
        log_fh.write(f"$ {' '.join(command)}\n")
        try:
            result = subprocess.run(
                command,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.output or "") if isinstance(exc.output, str) else ""
            log_fh.write(output + f"\n[env_setup] timed out after {timeout_s}s\n")
            return 124, output
        log_fh.write(result.stdout or "")
        return result.returncode, result.stdout or ""


def _benchmark_support_install_command(python_path: Path) -> list[str]:
    """Return the command that installs benchmark-only support packages."""
    return ["uv", "pip", "install", "--python", str(python_path), *BENCHMARK_SUPPORT_PACKAGES]


def _benchmark_support_local_project_commands(source: Path, python_path: Path) -> list[list[str]]:
    """Return commands that install local commit packages needed by benchmarks."""
    commands: list[list[str]] = []
    for rel_path in BENCHMARK_SUPPORT_LOCAL_PROJECTS:
        project_dir = source / rel_path
        if not project_dir.exists():
            continue
        if not (project_dir / "pyproject.toml").exists() and not (project_dir / "setup.py").exists():
            continue
        commands.append(["uv", "pip", "install", "--python", str(python_path), "--editable", str(project_dir)])
    return commands


def with_arm_libgomp_preload(env: dict[str, str]) -> dict[str, str]:
    """Return ``env`` with Isaac Sim's required ARM libgomp preload when needed."""
    if platform.machine().lower() not in ("aarch64", "arm64") or not _ARM_LIBGOMP_PATH.exists():
        return env
    preload = str(_ARM_LIBGOMP_PATH)
    current = env.get("LD_PRELOAD", "")
    entries = [entry for entry in current.split(":") if entry]
    if preload not in entries:
        entries.append(preload)
    return {**env, "LD_PRELOAD": ":".join(entries)}


def with_omniverse_eula_acceptance(env: dict[str, str]) -> dict[str, str]:
    """Return ``env`` with Omniverse Kit's non-interactive EULA acceptance set.

    Importing ``isaacsim`` (as the post-install import verification does) bootstraps
    Omniverse Kit, which on a fresh environment prompts for EULA acceptance on stdin.
    The reconstruct subprocess has no TTY, so the prompt reads EOF and the kernel
    fails to bootstrap, which would otherwise be misclassified as a
    ``runtime_incompatible`` skip. Setting these makes Kit accept the EULA
    non-interactively. Existing values win so a caller can still override.
    """
    accepted = dict(env)
    accepted.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
    accepted.setdefault("ACCEPT_EULA", "Y")
    return accepted


def with_uv_download_tuning(env: dict[str, str], *, cache_root: Path) -> dict[str, str]:
    """Return ``env`` tuned so heavy Isaac Sim/CUDA wheel downloads are robust and reused.

    Three settings, all important for per-commit reconstruction:

    * ``UV_CACHE_DIR`` is pinned under the shared env-cache root so the multi-GB
      Isaac Sim/CUDA wheels download once and are reused across commits. This
      matters most for ``docker-reconstruct``, where each candidate runs in a
      ``--rm`` container whose default ``~/.cache/uv`` would be discarded, forcing
      a full re-download (and re-exposing download flakiness) every commit.
    * ``UV_PYTHON_INSTALL_DIR`` keeps uv-managed interpreters under the same shared
      root. Otherwise a Docker-created venv links to ``~/.local/share/uv/python``
      inside the ephemeral container and becomes unusable when that container exits.
    * ``UV_HTTP_TIMEOUT`` is raised because some CUDA wheels (e.g. the aarch64
      ``nvidia-cusolver`` build) are large enough to exceed uv's short default
      request timeout on slower mirrors, which otherwise surfaces as a spurious
      ``install_failed``.

    Existing values in ``env`` win, so a caller can still override any setting.
    """
    tuned = dict(env)
    tuned.setdefault("UV_CACHE_DIR", str(Path(cache_root) / "uv-cache"))
    tuned.setdefault("UV_PYTHON_INSTALL_DIR", str(Path(cache_root) / "uv-python"))
    tuned.setdefault("UV_HTTP_TIMEOUT", "600")
    return tuned


def ensure_env(
    stack: StackSpec,
    source_dir: Path | str,
    cache_root: Path | str,
    *,
    install_scope: str = DEFAULT_INSTALL_SCOPE,
    python_version: str | None = None,
    timeout_s: int = 5400,
    force: bool = False,
) -> EnvHandle:
    """Build (or reuse) an isolated environment reproducing a commit's pinned stack.

    Creates a ``uv`` virtual environment dedicated to the commit and installs the
    commit's own clone via ``./isaaclab.sh -i <install_scope>`` so both the pinned
    runtime stack and the commit's code are reproduced. A ``built.ok`` sentinel
    lets a re-measured commit reuse its environment. The heavy downloads are
    amortized across commits by uv's global cache and hardlinking.

    Args:
        stack: The resolved stack for the commit (provides the commit SHA and the
            recorded versions).
        source_dir: Path to the commit's checked-out clone (contains ``isaaclab.sh``).
        cache_root: Root directory under which per-commit environments and install
            logs are stored.
        install_scope: ``./isaaclab.sh -i`` scope string; defaults to
            :data:`DEFAULT_INSTALL_SCOPE`.
        python_version: Interpreter ``X.Y`` for the venv; defaults to the commit's
            recorded Python version.
        timeout_s: Per-command timeout for the venv creation, install, and verify.
        force: If True, discard any cached environment for this commit and rebuild
            from scratch (used by a recovery reinstall after a suspected bad build).

    Returns:
        An :class:`EnvHandle` pointing at the environment's Python.

    Raises:
        EnvSkip: If the venv cannot be created, the install fails, or the resulting
            environment fails the import verification.
    """
    source = Path(source_dir)
    root = Path(cache_root)
    env_dir = root / "envs" / stack.commit_sha[:12]
    python_path = env_dir / "bin" / "python"
    sentinel = env_dir / "built.ok"
    if force and env_dir.exists():
        _progress(f"discarding cached environment for {stack.commit_sha[:12]}")
        shutil.rmtree(env_dir)

    can_reuse = sentinel.exists() and python_path.exists()
    if env_dir.exists() and not can_reuse:  # clear any partial build
        _progress(f"removing incomplete environment for {stack.commit_sha[:12]}")
        shutil.rmtree(env_dir)
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = root / "logs" / f"install-{stack.commit_sha[:12]}.log"
    install_env = with_omniverse_eula_acceptance(
        with_uv_download_tuning(with_arm_libgomp_preload({**os.environ, "VIRTUAL_ENV": str(env_dir)}), cache_root=root)
    )
    install_env.pop("CONDA_PREFIX", None)  # ensure isaaclab.sh targets the new venv

    def ensure_benchmark_support() -> None:
        """Install benchmark-only support into fresh or reused environments."""
        _progress(f"installing benchmark support for {stack.commit_sha[:12]}")
        rc, out = _run_logged(
            _benchmark_support_install_command(python_path),
            cwd=source,
            env=install_env,
            log_path=log_path,
            timeout_s=timeout_s,
        )
        if rc != 0:
            raise EnvSkip("install_failed", f"benchmark support packages: {_tail(out)}")

        for command in _benchmark_support_local_project_commands(source, python_path):
            rc, out = _run_logged(
                command,
                cwd=source,
                env=install_env,
                log_path=log_path,
                timeout_s=timeout_s,
            )
            if rc != 0:
                project = Path(command[-1]).name
                raise EnvSkip("install_failed", f"benchmark support project {project}: {_tail(out)}")

    def verify_imports() -> None:
        _progress(f"verifying reconstructed imports for {stack.commit_sha[:12]}")
        rc, out = _run_logged(
            [
                str(source / "isaaclab.sh"),
                "-p",
                "-c",
                "import hydra; import isaaclab; import isaaclab_tasks; import isaacsim; print('ENV_OK')",
            ],
            cwd=source,
            env=install_env,
            log_path=log_path,
            timeout_s=timeout_s,
        )
        if rc != 0 or "ENV_OK" not in out:
            raise EnvSkip("runtime_incompatible", f"import verification failed: {_tail(out)}")

    interpreter = python_version or stack.python_version
    if sentinel.exists() and python_path.exists():
        _progress(f"reusing environment {env_dir}")
        ensure_benchmark_support()
        verify_imports()
        _progress(f"environment ready for {stack.commit_sha[:12]} (reused)")
        return EnvHandle(str(python_path), str(env_dir), stack.stack_hash, stack.isaacsim, reused=True)

    _progress(f"creating Python {interpreter} environment for {stack.commit_sha[:12]}")
    rc, out = _run_logged(
        ["uv", "venv", str(env_dir), "--python", interpreter],
        cwd=source,
        env=with_uv_download_tuning({**os.environ}, cache_root=root),
        log_path=log_path,
        timeout_s=timeout_s,
    )
    if rc != 0:
        raise EnvSkip("install_failed", f"uv venv failed: {_tail(out)}")

    _progress(f"installing pinned runtime stack for {stack.commit_sha[:12]}")
    rc, out = _run_logged(
        [str(source / "isaaclab.sh"), "-i", install_scope],
        cwd=source,
        env=install_env,
        log_path=log_path,
        timeout_s=timeout_s,
    )
    if rc != 0:
        raise EnvSkip(_classify_install_failure(out), f"isaaclab.sh -i {install_scope}: {_tail(out)}")

    ensure_benchmark_support()
    verify_imports()

    sentinel.write_text("ok\n", encoding="utf-8")
    _progress(f"environment ready for {stack.commit_sha[:12]} (fresh)")
    return EnvHandle(str(python_path), str(env_dir), stack.stack_hash, stack.isaacsim, reused=False)


def _main(argv: list[str] | None = None) -> int:
    """Standalone CLI for resolving a stack or reconstructing one commit's env."""
    parser = argparse.ArgumentParser(description="Resolve a commit's pinned stack or reconstruct its environment.")
    parser.add_argument("--repo_root", default=".", help="Repository root for git lookups.")
    sub = parser.add_subparsers(dest="command", required=True)

    resolve = sub.add_parser("resolve-stack", help="Print the pinned stack for a commit.")
    resolve.add_argument("--commit", required=True)

    build = sub.add_parser("ensure-env", help="Reconstruct one commit's environment.")
    build.add_argument("--commit", required=True)
    build.add_argument("--source_dir", required=True, help="Checked-out clone of the commit.")
    build.add_argument("--cache_root", required=True)
    build.add_argument("--install_scope", default=DEFAULT_INSTALL_SCOPE)

    args = parser.parse_args(argv)
    stack = resolve_stack(args.repo_root, args.commit)
    if args.command == "resolve-stack":
        print(json.dumps(stack.to_json(), indent=2, sort_keys=True))
        return 0

    try:
        handle = ensure_env(stack, args.source_dir, args.cache_root, install_scope=args.install_scope)
    except EnvSkip as skip:
        print(json.dumps({"status": "skip", "category": skip.category, "detail": skip.detail}, indent=2))
        return 3
    print(json.dumps({"status": "ok", **handle.to_json()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
