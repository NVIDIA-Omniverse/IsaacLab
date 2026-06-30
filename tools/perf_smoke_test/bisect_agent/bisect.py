"""
bisect.py — CLI entry point for the bisect agent.

Usage examples:

  # Full run (LLM orchestrator + diagnosis):
  python bisect.py \
      --good abc1234 --bad def5678 \
      --task Isaac-Velocity-Flat-G1-Direct \
      --backend newton

  # Deterministic stages only (no LLM):
  python bisect.py \
      --good abc1234 --bad def5678 \
      --task Isaac-Velocity-Flat-G1-Direct \
      --backend newton \
      --no-llm

  # Dev mode end-to-end test (no Docker / GPU):
  python bisect.py \
      --good abc1234 --bad def5678 \
      --task Isaac-Cartpole-Direct \
      --backend newton \
      --dev \
      --dev-perf-map tests/dev_perf_map.json \
      --output-dir runs/test_e2e \
      --no-llm

  # Override LLM model and API base URL:
  python bisect.py \
      --good abc1234 --bad def5678 \
      --task Isaac-Velocity-Flat-G1-Direct \
      --backend newton \
      --model claude-sonnet-4-6 \
      --base-url https://api.anthropic.com/v1
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# IMPORTANT: This file is named bisect.py, which shadows the stdlib 'bisect'
# module.  Pre-load the real stdlib bisect into sys.modules NOW (before any
# other import can trigger statistics→random→bisect and hit the shadowed file).
# ---------------------------------------------------------------------------
import sys as _sys
import importlib.util as _ilu
_bisect_spec = _ilu.find_spec("bisect")
if _bisect_spec is not None and _bisect_spec.origin != __file__:
    # Only pre-load if the spec points to a DIFFERENT file (i.e. the real stdlib)
    _bisect_mod = _ilu.module_from_spec(_bisect_spec)
    _bisect_spec.loader.exec_module(_bisect_mod)  # type: ignore[union-attr]
    _sys.modules["bisect"] = _bisect_mod
else:
    # We ARE the file being picked up as 'bisect' — load from stdlib location.
    import importlib.machinery as _ilm
    import os as _os
    _stdlib_dir = _os.path.dirname(_os.path.realpath(_os.__file__))
    _bisect_path = _os.path.join(_stdlib_dir, "bisect.py")
    if not _os.path.exists(_bisect_path):
        # CPython 3.9+: bisect is in lib-dynload as C extension
        for _d in _sys.path:
            _candidate = _os.path.join(_d, "bisect") if _d else None
            if _candidate and _os.path.exists(_candidate + ".cpython-312-x86_64-linux-gnu.so"):
                _bisect_path = _candidate + ".cpython-312-x86_64-linux-gnu.so"
                break
    if _os.path.exists(_bisect_path):
        _loader = _ilm.SourceFileLoader("bisect", _bisect_path)
        _bisect_spec2 = _ilu.spec_from_loader("bisect", _loader)  # type: ignore[arg-type]
        _bisect_mod2 = _ilu.module_from_spec(_bisect_spec2)  # type: ignore[arg-type]
        _loader.exec_module(_bisect_mod2)
        _sys.modules["bisect"] = _bisect_mod2
del _ilu, _bisect_spec

import argparse
import json
import logging
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure bisect_agent/ is on sys.path so sub-modules can be imported both
# when called as "python bisect.py" (already on path) and when invoked from
# another directory or as a module.
# ---------------------------------------------------------------------------
_AGENT_DIR = Path(__file__).resolve().parent
if str(_AGENT_DIR) not in sys.path:
    # Append rather than prepend so stdlib modules (e.g. 'bisect') are found
    # before this directory — avoids shadowing stdlib with bisect.py itself.
    sys.path.append(str(_AGENT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("bisect")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SHA_LABEL = 7   # shorter prefix used in human-readable names (run dir, CLI output)
_SHA_SHORT = 12  # longer prefix used in artifact subdirs (matches core modules)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(data, fh, indent=2)


def _read_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bisect.py",
        description="Automated commit bisection for IsaacLab performance regressions.",
    )
    p.add_argument("--good", required=True, help="Known-good commit SHA.")
    p.add_argument("--bad", required=True, help="Known-bad commit SHA.")
    p.add_argument("--task", required=True, help="IsaacLab task ID (e.g. Isaac-Velocity-Flat-G1-Direct).")
    p.add_argument("--backend", required=True, help="Backend key (e.g. newton).")
    p.add_argument(
        "--repo",
        default=None,
        help="Path to IsaacLab repo. Defaults to ../IsaacLab relative to bisect.py.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help=(
            "Output directory for all run artifacts. "
            "Auto-named runs/{task}_{backend}_{good[:7]}_{bad[:7]} if omitted."
        ),
    )
    p.add_argument(
        "--dev",
        action="store_true",
        help="Dev mode: no Docker/GPU; uses stub_benchmark.py with STUB_FPS_MEAN.",
    )
    p.add_argument(
        "--dev-perf-map",
        default=None,
        dest="dev_perf_map",
        help="Path to JSON file mapping SHA -> fps_mean for dev mode.",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        dest="no_llm",
        help="Skip LLM orchestrator and diagnosis; run deterministic stages only.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="LLM model override (e.g. claude-sonnet-4-6).",
    )
    p.add_argument(
        "--base-url",
        default=None,
        dest="base_url",
        help="LLM API base URL override (e.g. https://api.anthropic.com/v1).",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Resolve paths and load dev_perf_map.
    # ------------------------------------------------------------------
    repo_path: Path = (
        Path(args.repo).resolve()
        if args.repo
        else (_AGENT_DIR / ".." / ".." / "..").resolve()  # bisect_agent/ -> perf_smoke_test/ -> tools/ -> IsaacLab/
    )

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        auto_name = f"{args.task}_{args.backend}_{args.good[:_SHA_LABEL]}_{args.bad[:_SHA_LABEL]}"
        output_dir = (_AGENT_DIR / "runs" / auto_name).resolve()

    dev_perf_map: dict | None = None
    if args.dev_perf_map:
        dev_perf_map_path = Path(args.dev_perf_map).resolve()
        try:
            with dev_perf_map_path.open() as fh:
                dev_perf_map = json.load(fh)
            logger.info("Loaded dev_perf_map from %s (%d entries).", dev_perf_map_path, len(dev_perf_map))
        except FileNotFoundError:
            logger.error("--dev-perf-map file not found: %s", dev_perf_map_path)
            sys.exit(1)
        except json.JSONDecodeError as exc:
            logger.error("--dev-perf-map is not valid JSON: %s", exc)
            sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # ------------------------------------------------------------------
    # 2. Write run_config.json and initial status.json.
    # ------------------------------------------------------------------
    run_config: dict = {
        "good_sha": args.good,
        "bad_sha": args.bad,
        "task_id": args.task,
        "backend": args.backend,
        "repo_path": str(repo_path),
        "output_dir": str(output_dir),
        "dev_mode": args.dev,
        "dev_perf_map_path": args.dev_perf_map,
        "no_llm": args.no_llm,
        "model": args.model,
        "base_url": args.base_url,
        "started_at": _now_iso(),
    }
    _write_json(output_dir / "run_config.json", run_config)

    # Write env snapshot as first audit entry so every run is reproducible.
    _env_entry: dict = {
        "ts": run_config["started_at"],
        "step": "env_snapshot",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "bisect_agent_dir": str(_AGENT_DIR),
        "env_vars": {k: os.environ.get(k, "") for k in (
            "BISECT_LLM_MODEL", "BISECT_LLM_BASE_URL",
            "CUDA_VISIBLE_DEVICES", "STUB_FPS_MEAN",
        )},
        "run_config": {
            "good_sha": args.good,
            "bad_sha": args.bad,
            "task_id": args.task,
            "backend": args.backend,
            "dev_mode": args.dev,
        },
    }
    _audit_log = output_dir / "audit_log.jsonl"
    with _audit_log.open("a") as _fh:
        _fh.write(json.dumps(_env_entry) + "\n")

    status: dict = {
        "phase": "init",
        "status": "running",
        "progress": "starting",
        "last_update": _now_iso(),
    }
    _write_json(output_dir / "status.json", status)

    # ------------------------------------------------------------------
    # 3. Import core modules — fail fast with helpful messages.
    # ------------------------------------------------------------------
    try:
        from core import runner as runner_mod
    except ImportError as exc:
        logger.error("Failed to import core.runner: %s", exc)
        logger.error("Ensure bisect_agent/core/runner.py exists and its dependencies are installed.")
        sys.exit(1)

    try:
        from core.grounding import run_grounding
    except ImportError as exc:
        logger.error("Failed to import core.grounding.run_grounding: %s", exc)
        sys.exit(1)

    try:
        from core.bisector import run_bisect
    except ImportError as exc:
        logger.error("Failed to import core.bisector.run_bisect: %s", exc)
        sys.exit(1)

    try:
        from core import diagnosis as diagnosis_mod
    except ImportError as exc:
        logger.warning("core.diagnosis not available: %s", exc)
        diagnosis_mod = None  # type: ignore[assignment]

    try:
        from infra import commits as commits_mod
    except ImportError as exc:
        logger.error("Failed to import infra.commits: %s", exc)
        logger.error("Ensure bisect_agent/infra/commits.py exists.")
        sys.exit(1)

    try:
        from infra import container as container_mod  # noqa: F401 — imported for side-effect availability
    except ImportError as exc:
        logger.warning("infra.container not available: %s", exc)

    # Partial application: runner callable pre-filled with dev flags.
    def _run_commit(sha: str, task_id: str, backend: str, out_dir: Path, **kwargs) -> dict:
        kw = {"dev_mode": args.dev, "dev_perf_map": dev_perf_map}
        kw.update(kwargs)
        return runner_mod.run_commit(sha, task_id, backend, out_dir, **kw)

    # ------------------------------------------------------------------
    # 4. --no-llm path: deterministic stages only.
    # ------------------------------------------------------------------
    if args.no_llm:
        # ---- Stage 1: Grounding ----------------------------------------
        print("\n=== Stage 1: Grounding ===")
        status = {"phase": "grounding", "status": "running", "progress": "running good/bad experiments", "last_update": _now_iso()}
        _write_json(output_dir / "status.json", status)

        grounding_result = run_grounding(
            good_sha=args.good,
            bad_sha=args.bad,
            task_id=args.task,
            backend=args.backend,
            run_dir=output_dir,
            runner_run_commit=_run_commit,
            dev_mode=args.dev,
            dev_perf_map=dev_perf_map,
        )

        print(f"  separated   : {grounding_result['separated']}")
        print(f"  verdict     : {grounding_result['verdict']}")
        print(f"  kpi_deltas  : {grounding_result.get('kpi_deltas', {})}")
        if grounding_result.get("note"):
            print(f"  note        : {grounding_result['note']}")

        # ---- Stage 2: Enumerate commits --------------------------------
        print("\n=== Stage 2: Commit Enumeration ===")
        status = {"phase": "commits", "status": "running", "progress": "enumerating commits", "last_update": _now_iso()}
        _write_json(output_dir / "status.json", status)

        commits_path = output_dir / "commits.json"
        if commits_path.exists():
            logger.info("commits.json already exists — loading from cache.")
            with commits_path.open() as fh:
                commits: list[dict] = json.load(fh)
        else:
            commits = commits_mod.enumerate_commits(
                good_sha=args.good,
                bad_sha=args.bad,
                repo_path=repo_path,
            )
            _write_json(commits_path, commits)

        print(f"  commits in range: {len(commits)}")

        # ---- Stage 3: Bisect -------------------------------------------
        print("\n=== Stage 3: Binary Search ===")
        status = {"phase": "bisect", "status": "running", "progress": f"bisecting {len(commits)} commits", "last_update": _now_iso()}
        _write_json(output_dir / "status.json", status)

        bisect_result = run_bisect(
            commits=commits,
            grounding_result=grounding_result,
            task_id=args.task,
            backend=args.backend,
            run_dir=output_dir,
            runner_run_commit=_run_commit,
            dev_mode=args.dev,
            dev_perf_map=dev_perf_map,
        )

        print(f"  first_bad_sha   : {bisect_result['first_bad_sha']}")
        print(f"  prev_good_sha   : {bisect_result.get('prev_good_sha')}")
        print(f"  commits_tested  : {bisect_result['commits_tested']}")
        print(f"  confidence      : {bisect_result['confidence']}")

        status = {"phase": "done_no_llm", "status": "done", "progress": "bisect complete; diagnosis skipped (--no-llm)", "last_update": _now_iso()}
        _write_json(output_dir / "status.json", status)

        print(
            "\nDeterministic stages complete. "
            "Re-run without --no-llm to perform LLM diagnosis."
        )

    # ------------------------------------------------------------------
    # 5. LLM orchestrator path.
    # ------------------------------------------------------------------
    else:
        try:
            from infra.llm_client import LLMClient
        except ImportError as exc:
            logger.error("Failed to import infra.llm_client.LLMClient: %s", exc)
            logger.error(
                "Ensure bisect_agent/infra/llm_client.py exists and the 'openai' "
                "package is installed (pip install openai)."
            )
            sys.exit(1)

        try:
            from orchestrator import run_orchestrator
        except ImportError as exc:
            logger.error("Failed to import orchestrator.run_orchestrator: %s", exc)
            logger.error("Ensure bisect_agent/orchestrator.py exists.")
            sys.exit(1)

        llm_client = LLMClient(
            model=args.model,
            base_url=args.base_url,
        )

        status = {"phase": "orchestrator", "status": "running", "progress": "LLM orchestrator started", "last_update": _now_iso()}
        _write_json(output_dir / "status.json", status)

        import functools
        from core.grounding import run_grounding as _run_grounding
        from core.bisector import run_bisect as _run_bisect
        from core.diagnosis import run_diagnosis as _run_diagnosis
        from infra.commits import enumerate_commits as _enum_commits
        from infra.commits import fetch_diff as _fetch_diff

        run_orchestrator(
            run_config=run_config,
            run_dir=output_dir,
            llm_client=llm_client,
            runner_run_commit=_run_commit,
            commits_enumerate=functools.partial(_enum_commits, repo_path=repo_path),
            commits_fetch_diff=functools.partial(_fetch_diff, repo_path=repo_path),
            grounding_run=_run_grounding,
            bisect_run=_run_bisect,
            diagnosis_run=_run_diagnosis,
            dev_mode=args.dev,
            dev_perf_map=dev_perf_map,
            repo_path=repo_path,
        )

    # ------------------------------------------------------------------
    # 6. Final summary + run_summary.json
    # ------------------------------------------------------------------
    import glob

    finished_at = _now_iso()
    bisect_result_path = output_dir / "bisect_result.json"
    bisect_result = _read_json(bisect_result_path) if bisect_result_path.exists() else {}

    grounding_runs = len(list(output_dir.glob("grounding/*/run_result.json")))
    bisect_runs    = len(list(output_dir.glob("bisect/*/run_result.json")))
    diag_runs      = len(list(output_dir.glob("diagnosis/*/run_result.json")))

    run_summary = {
        "cli_invocation": " ".join(
            ["python bisect.py"]
            + [f"--good {args.good}", f"--bad {args.bad}",
               f"--task {args.task}", f"--backend {args.backend}"]
            + (["--dev"] if args.dev else [])
            + ([f"--dev-perf-map {args.dev_perf_map}"] if args.dev_perf_map else [])
            + (["--no-llm"] if args.no_llm else [])
        ),
        "started_at":  run_config.get("started_at"),
        "finished_at": finished_at,
        "good_sha":    args.good,
        "bad_sha":     args.bad,
        "task_id":     args.task,
        "backend":     args.backend,
        "first_bad_sha": bisect_result.get("first_bad_sha"),
        "prev_good_sha": bisect_result.get("prev_good_sha"),
        "confidence":    bisect_result.get("confidence"),
        "bench_runs": {
            "grounding":  grounding_runs,
            "bisect":     bisect_runs,
            "diagnosis":  diag_runs,
            "total":      grounding_runs + bisect_runs + diag_runs,
        },
        "output_dir": str(output_dir),
        "audit_log":  str(output_dir / "audit_log.jsonl"),
        "report":     str(output_dir / "report" / "report.md"),
    }
    _write_json(output_dir / "run_summary.json", run_summary)

    print("\n=== Run Complete ===")
    print(f"  output_dir : {output_dir}")
    if bisect_result:
        print(f"  first_bad  : {bisect_result.get('first_bad_sha', 'unknown')}")
        print(f"  confidence : {bisect_result.get('confidence', '?')}")
    else:
        print("  first_bad  : (bisect_result.json not found)")
    print(f"  bench_runs : grounding={grounding_runs}, bisect={bisect_runs}, diagnosis={diag_runs}")

    report_path = output_dir / "report" / "report.md"
    if report_path.exists():
        print(f"  report     : {report_path}")
    else:
        print("  report     : (not generated — run without --no-llm for LLM diagnosis)")
    print(f"  summary    : {output_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
