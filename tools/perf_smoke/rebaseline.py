# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure tasks N times and (optionally) refresh the rolling in-tree baseline.

One measurement path serves two jobs:

* **Variance study (report mode, default):** run each gate task ``--repeat`` times,
  report robust per-task stats (median / CV / MAD / min / max), and show how the
  spread compares to the configured thresholds. Use this to justify thresholds to
  reviewers and to confirm baselines transfer to the current environment.
* **Rolling re-baseline (``--apply``):** turn that same window of recent runs into
  updated ``baseline.json`` values (``baseline_fps`` = window median; warn/block
  from window CV). Values stay **in-tree** and the CI workflow opens a *PR* with
  the diff -- the change is always reviewed.

Boiling-frog guard
------------------
A rolling baseline must not quietly absorb a real regression. Any task whose new
median would drop the baseline by more than ``--soft-drop-pct`` is **flagged for
review** (applied, but loudly marked in the report / PR body). A drop beyond
``--hard-drop-pct`` is **refused** (old value kept) unless ``--force`` -- a drop
that large is almost certainly a regression, not drift.

Reuses :mod:`run_perf_gate` for command building and :mod:`check_perf_regression`
for FPS extraction, so a measurement here is identical to a gate measurement.

Examples::

    # variance study: 5 reps of all baseline tasks, report only
    ./isaaclab.sh -p tools/perf_smoke/rebaseline.py --repeat 5

    # rolling re-baseline: write proposed values (CI then opens a PR)
    ./isaaclab.sh -p tools/perf_smoke/rebaseline.py --repeat 5 --apply
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import check_perf_regression as cpr  # noqa: E402
import run_perf_gate as gate  # noqa: E402


def _baseline_tasks(baseline: dict) -> list[str]:
    """Return the non-metadata task keys from a baseline document, in order."""
    return [k for k in baseline if not k.startswith("_")]


def _measure_fps(task_id: str, run_dir: Path, warmup: int, num_frames: int | None) -> float | None:
    """Extract the post-warm-up steady FPS (the gating metric, D6) from a run dir.

    ``task_id`` is the gym task the benchmark was launched with (the result-file name),
    which for a variant key like ``Isaac-Cartpole-v0@newton`` is just ``Isaac-Cartpole-v0``.
    """
    pattern = cpr.DEFAULT_GLOB_TEMPLATE.format(task=task_id)
    try:
        result_path = cpr._resolve_results(str(run_dir), pattern, allow_multiple=True)
        return cpr.steady_fps(cpr._load_result(result_path), warmup, num_frames)
    except cpr.CompareError:
        return None


def _window_stats(samples: list[float]) -> dict | None:
    """Robust summary of a measurement window. ``None`` when there are no samples."""
    if not samples:
        return None
    n = len(samples)
    median = statistics.median(samples)
    mean = statistics.fmean(samples)
    # Sample CV (needs >= 2 points); single-sample windows report 0 spread.
    cv_pct = (statistics.stdev(samples) / mean * 100.0) if n >= 2 and mean > 0 else 0.0
    mad = statistics.median([abs(s - median) for s in samples])
    return {
        "n": n,
        "median": median,
        "mean": mean,
        "cv_pct": cv_pct,
        "mad": mad,
        "min": min(samples),
        "max": max(samples),
        "samples": samples,
    }


def _proposed_entry(stats: dict) -> dict:
    """Map window stats to baseline fields (median anchor; CV-derived thresholds)."""
    cv = stats["cv_pct"]
    return {
        "baseline_fps": round(stats["median"], 1),
        "warn_pct": round(max(3.0 * cv, 5.0), 1),
        "max_regression_pct": round(max(6.0 * cv, 10.0), 1),
        "cv_pct": round(cv, 2),
        "n_runs": stats["n"],
    }


def measure_task(
    task: str,
    baseline: dict,
    out_root: Path,
    repeat: int,
    cache_dir: str | None,
    seed_override: int | None = None,
    tag: str = "",
) -> dict | None:
    """Run ``task`` ``repeat`` times and return its window stats (or ``None``).

    The returned stats also carry the per-run ``wall`` samples so ``--apply`` can
    append both FPS and wall-clock to the rolling-window store.

    ``seed_override`` replaces the baseline seed (used by the seed-sweep study to vary
    the random scene); ``tag`` namespaces the run dirs so concurrent seeds/reps don't
    collide.
    """
    try:
        cfg = gate._task_run_config(baseline, task)
    except KeyError as e:
        print(f"[rebaseline] {task}: {e} -> skipped", flush=True)
        return None
    if seed_override is not None:
        cfg["seed"] = seed_override
    entry = baseline.get(task, {})
    task_id = cfg.get("task_id", task)  # gym id used for the result-file glob (vs the @variant gate key)
    warmup = int(entry.get("warmup_frames", cpr.DEFAULT_WARMUP_FRAMES))
    num_frames = entry.get("num_frames")
    num_frames = int(num_frames) if isinstance(num_frames, (int, float)) else None
    samples: list[float] = []
    walls: list[float] = []
    for rep in range(1, repeat + 1):
        run_dir = out_root / task / f"{tag}rep{rep}"
        print(f"\n[rebaseline] === {task} {tag}rep {rep}/{repeat} (seed={cfg['seed']}) ===", flush=True)
        wall = gate._run_benchmark(task, cfg, run_dir, retries=1, dry_run=False, cache_dir=cache_dir)
        if wall is None:
            print(f"[rebaseline] {task} rep {rep}: run failed -> dropped from window", flush=True)
            continue
        fps = _measure_fps(task_id, run_dir, warmup, num_frames)
        if fps is None:
            print(f"[rebaseline] {task} rep {rep}: could not read FPS -> dropped", flush=True)
            continue
        print(f"[rebaseline] {task} rep {rep}: {fps:.0f} FPS ({wall:.0f}s)", flush=True)
        samples.append(fps)
        walls.append(wall)
    stats = _window_stats(samples)
    if stats is not None:
        stats["walls"] = walls
    return stats


def _append_window(history_dir: Path, task: str, gpu_key: str, stats: dict, cap: int = 20) -> int:
    """Append this study's samples to the rolling-window store; prune to ``cap``.

    This is the orphan-branch update in the doc's model: each study contributes
    its runs to ``<history-dir>/<task>__<gpu>.json``, oldest dropped past ``cap``.
    Returns the resulting window length.
    """
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / f"{task}__{gpu_key}.json".replace(" ", "_")
    store = {"task": task, "gpu": gpu_key, "window": cap, "samples": []}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(existing, dict) and isinstance(existing.get("samples"), list):
            store = existing
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    walls = stats.get("walls") or [None] * len(stats["samples"])
    for fps, wall in zip(stats["samples"], walls):
        store["samples"].append({"fps": round(fps, 1), "wall_s": wall, "ts": now, "source": "rebaseline"})
    store["samples"] = store["samples"][-cap:]
    store["window"] = cap
    path.write_text(json.dumps(store, indent=2) + "\n", encoding="utf-8")
    return len(store["samples"])


def _current_baseline_fps(baseline: dict, task: str, gpu_key: str) -> float | None:
    entry = baseline.get(task, {}).get("per_gpu", {}).get(gpu_key)
    if isinstance(entry, dict) and "baseline_fps" in entry:
        return float(entry["baseline_fps"])
    return None


def _emit_summary(line: str) -> None:
    """Print and append to the CI step summary when available."""
    print(line)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


def _seed_study(args: argparse.Namespace, baseline: dict, out_root: Path, tasks: list[str]) -> int:
    """Report-only seed-sensitivity study (never touches the window or baseline).

    For each task it compares two coefficients of variation:

    * **same-seed CV** -- ``--repeat`` reps at the baseline (gate) seed: the run-to-run
      noise the gate already lives with;
    * **cross-seed CV** -- one run per ``--seeds`` value: how much the random scene alone
      moves FPS.

    If cross-seed CV is in the same ballpark as same-seed CV, FPS is seed-insensitive and
    hard-coding the gate seed costs nothing. If it is much larger, the task's band should
    widen (or the gate should average a few seeds).
    """
    _emit_summary("\n## Perf seed-sensitivity study\n")
    _emit_summary(
        f"GPU key `{args.gpu_key}`; same-seed = {args.repeat} reps @ baseline seed, cross-seed = {args.seeds}.\n"
    )
    _emit_summary(
        "| Task | gate seed | same-seed n / median / CV% | cross-seed n / median / CV% | cross/same | verdict |"
    )
    _emit_summary("|---|---:|---|---|---:|---|")
    for task in tasks:
        gate_seed = gate._task_run_config(baseline, task).get("seed", 42)
        same = measure_task(task, baseline, out_root, args.repeat, args.cache_dir, tag="sameseed_")
        cross_fps: list[float] = []
        for seed in args.seeds:
            s = measure_task(task, baseline, out_root, 1, args.cache_dir, seed_override=seed, tag=f"seed{seed}_")
            if s is not None:
                cross_fps.append(s["median"])
        cross = _window_stats(cross_fps)
        if same is None or cross is None:
            _emit_summary(
                f"| {task} | {gate_seed} | {'—' if same is None else 'ok'} | measurement failed | — | ⚠️skip |"
            )
            continue
        ratio = cross["cv_pct"] / same["cv_pct"] if same["cv_pct"] > 0 else float("inf")
        # Seed is "safe to fix" when the scene adds no more spread than ordinary run noise
        # (allow a small absolute floor so near-zero same-seed CVs don't explode the ratio).
        safe = cross["cv_pct"] <= max(same["cv_pct"] * 1.5, same["cv_pct"] + 0.3)
        verdict = "✅ seed-insensitive" if safe else "⚠️ widen band / avg seeds"
        _emit_summary(
            f"| {task} | {gate_seed} | {same['n']} / {same['median']:.0f} / {same['cv_pct']:.2f} | "
            f"{cross['n']} / {cross['median']:.0f} / {cross['cv_pct']:.2f} | {ratio:.2f}x | {verdict} |"
        )
    _emit_summary("\n_Report only: the seed sweep is never appended to the window or baseline._")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--tasks", nargs="+", default=None, help="Tasks to measure (default: all baseline tasks).")
    parser.add_argument("--baseline", default=str(_THIS_DIR / "baseline.json"), help="Path to baseline.json.")
    parser.add_argument("--repeat", type=int, default=5, help="Runs per task in the window (default 5).")
    parser.add_argument(
        "--output-dir", default=str(gate._REPO_ROOT / "perf-output-rebaseline"), help="Run output root."
    )
    parser.add_argument("--cache-dir", default=None, help="Optional warm JIT-cache dir (see run_perf_gate.py).")
    parser.add_argument("--gpu-key", default="NVIDIA L40S", help="Baseline per-GPU key to read/update.")
    parser.add_argument(
        "--history-dir", default=str(_THIS_DIR / "perf_history"), help="Rolling-window store to append to on --apply."
    )
    parser.add_argument("--window-cap", type=int, default=20, help="Max samples kept per task in the window.")
    parser.add_argument("--apply", action="store_true", help="Append to the window + refresh baseline.json fallback.")
    parser.add_argument("--soft-drop-pct", type=float, default=5.0, help="Drops beyond this are flagged for review.")
    parser.add_argument(
        "--hard-drop-pct", type=float, default=15.0, help="Drops beyond this are refused (unless --force)."
    )
    parser.add_argument("--force", action="store_true", help="Apply even hard-limit drops.")
    parser.add_argument("--stats-out", default=None, help="Optional path to write the raw window stats JSON.")
    parser.add_argument("--from-stats", default=None, help="Reuse a previous --stats-out window instead of measuring.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Seed-sensitivity study (report-only): measure same-seed CV (--repeat reps at the baseline seed) "
        "vs cross-seed CV (one run per given seed). Justifies hard-coding the gate seed.",
    )
    args = parser.parse_args(argv)

    baseline_path = Path(args.baseline).resolve()
    baseline = gate._load_baseline(baseline_path)
    out_root = Path(args.output_dir).resolve()
    tasks = args.tasks or _baseline_tasks(baseline)

    if args.seeds:
        return _seed_study(args, baseline, out_root, tasks)

    all_stats: dict[str, dict] = {}
    if args.from_stats:
        # Apply a window measured earlier (no GPU): measure once, review/apply later.
        cached = json.loads(Path(args.from_stats).read_text(encoding="utf-8"))
        all_stats = {t: cached[t] for t in tasks if t in cached}
    else:
        for task in tasks:
            stats = measure_task(task, baseline, out_root, args.repeat, args.cache_dir)
            if stats is not None:
                all_stats[task] = stats

    if args.stats_out:
        Path(args.stats_out).write_text(json.dumps(all_stats, indent=2), encoding="utf-8")

    # --- report -------------------------------------------------------------
    _emit_summary("\n## Perf baseline window study\n")
    source = "cached window" if args.from_stats else f"{args.repeat} run(s)/task"
    _emit_summary(f"GPU key `{args.gpu_key}`, {source}.\n")
    _emit_summary("| Task | n | median FPS | CV% | MAD | min | max | current | Δ vs current |")
    _emit_summary("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    flagged: list[str] = []
    refused: list[str] = []
    proposals: dict[str, dict] = {}
    for task, stats in all_stats.items():
        prop = _proposed_entry(stats)
        cur = _current_baseline_fps(baseline, task, args.gpu_key)
        delta = ((prop["baseline_fps"] - cur) / cur * 100.0) if cur else None
        delta_str = f"{delta:+.2f}%" if delta is not None else "—"
        cur_str = f"{cur:.0f}" if cur else "—"
        guard = ""
        if delta is not None and delta < -args.hard_drop_pct and not args.force:
            guard = " ❌refused"
            refused.append(task)
        elif delta is not None and delta < -args.soft_drop_pct:
            guard = " ⚠️review"
            flagged.append(task)
        proposals[task] = prop
        _emit_summary(
            f"| {task} | {stats['n']} | {stats['median']:.0f} | {stats['cv_pct']:.2f} | "
            f"{stats['mad']:.1f} | {stats['min']:.0f} | {stats['max']:.0f} | {cur_str} | {delta_str}{guard} |"
        )

    if flagged:
        _emit_summary(f"\n⚠️ **Flagged for review (drop > {args.soft_drop_pct}%):** {', '.join(flagged)}")
    if refused:
        _emit_summary(
            f"\n❌ **Refused (drop > {args.hard_drop_pct}%, likely a real regression):** {', '.join(refused)}"
        )

    if not args.apply:
        _emit_summary("\n_Report only. Re-run with `--apply` to write these into baseline.json._")
        return 0

    # --- apply --------------------------------------------------------------
    # Append to the rolling window (primary store) and refresh the static
    # fallback in baseline.json so both track reality.
    history_dir = Path(args.history_dir).resolve()
    for task, prop in proposals.items():
        if task in refused:
            print(f"[rebaseline] {task}: refused (hard-limit drop) -> keeping current value", flush=True)
            continue
        n_window = _append_window(history_dir, task, args.gpu_key, all_stats[task], cap=args.window_cap)
        print(f"[rebaseline] {task}: window now n={n_window}", flush=True)
        entry = baseline.setdefault(task, {}).setdefault("per_gpu", {}).setdefault(args.gpu_key, {})
        entry.update(prop)
    baseline_path.write_text(json.dumps(baseline, indent=4) + "\n", encoding="utf-8")
    applied = len(proposals) - len(refused)
    _emit_summary(f"\n✅ Appended to window `{history_dir}` and refreshed `{baseline_path.name}` ({applied} task(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
