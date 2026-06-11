# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare a benchmark result JSON against the perf-smoke baseline.

Used by the Phase 1 perf-smoke CI gate. Implements the test logic of
``ci-regression-gate-config-info.md``:

Verdict ``{PASS, WARN, BLOCK}`` (the doc's vocabulary), surfaced as a grep-able
``RESULT=<verdict>`` line and a rich ``$GITHUB_STEP_SUMMARY`` table:

* ``PASS``  -- measured KPI within threshold of the baseline center.
* ``WARN``  -- medium regression (between the warn and block bands), or a result
  that needed a retry. Advisory; does not block merge (exit ``0``).
* ``BLOCK`` -- a hard failure. ``kind=regression`` for a large KPI drop (exit
  ``1``); ``kind=hard_failure`` for a structural problem -- missing/malformed
  result, missing baseline, NaN/zero FPS, unknown GPU (exit ``2``). Both block.

KPIs (D-test-logic)
-------------------
* ``fps`` -- ``Mean Environment step effective FPS`` (primary), computed over the
  *post-warm-up* frames (D6): the per-frame effective-FPS array with the first
  ``warmup_frames`` dropped. This is the identical statistic the backend reports,
  just windowed, so it stays comparable to the rolling window.
* ``wall_s`` -- wall-clock seconds of the run (secondary signal; advisory).

Threshold & baseline strategy (D-threshold)
-------------------------------------------
Thresholds are computed *at test time* from a rolling window of historical
samples using a robust **median + MAD** estimator (tunable ``k``):

    center = median(window)
    spread = max(1.4826 * MAD(window), min_spread_pct/100 * center)
    WARN  when measured < center - k_warn  * spread
    BLOCK when measured < center - k_block * spread   (kind=regression)

The window lives in an orphan-branch history store (``--history-dir``; the local
stand-in is a plain directory, exactly like the warm-cache sidecar). Manual
overrides (k values, spread floor, pinned center, or skip) come from an in-tree
``baseline_overrides.json`` committed *with the PR*. When no window is available
yet (fresh store) the comparator falls back to the static ``baseline_fps`` and
``warn_pct`` / ``max_regression_pct`` carried in ``baseline.json`` so the gate
still produces a verdict.

Result formats
--------------
Both benchmark backends are accepted: OmniPerf (a dict) and JSON (a list of
phase objects, normalized into the dict shape on load). The JSON backend carries
the per-frame arrays the steady metric and debug KPIs need.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

EXIT_PASS = 0
EXIT_BLOCK = 1
EXIT_HARD_FAILURE = 2

METRIC_PHASE = "runtime"
# Scalar all-frames fallback (used only when the per-frame array is absent).
METRIC_NAME = "Mean Environment step effective FPS"
# Per-frame arrays (json backend), inside the "Step Frametimes" measurement.
FRAMETIMES_NAME = "Step Frametimes"
EFF_FPS_ARRAY = "Environment step effective FPS"
STEP_MS_ARRAY = "Environment step times"

# D6: frames dropped before computing the gating KPI (warm-up / first-touch /
# JIT pollution). Per-task value comes from baseline.json; this is the default.
DEFAULT_WARMUP_FRAMES = 2

# Robust-threshold defaults (D-threshold); overridable per task/gpu.
DEFAULT_K_WARN = 3.0
DEFAULT_K_BLOCK = 6.0
DEFAULT_MIN_SPREAD_PCT = 1.5  # spread floor as % of center (guards tiny windows)
MAD_TO_STD = 1.4826  # MAD -> std-equivalent for ~normal data
MIN_WINDOW = 3  # samples needed before the rolling estimator is trusted
OUTLIER_FACTOR = 2.0  # a step slower than 2x the steady median is an outlier
MAX_REPORTED_OUTLIERS = 8  # cap index/magnitude lists in the report
# Warm-up guardrail: if the first *kept* (post-warm-up) step is slower than this
# multiple of the steady median, warmup_frames is probably too small (pollution
# leaking into the KPI). Advisory only -- never changes the verdict.
WARMUP_GUARD_FACTOR = 3.0

DEFAULT_GLOB_TEMPLATE = "benchmark_non_rl_{task}*.json"
_JSON_NAME_PREFIX = "benchmark_non_rl"


class CompareError(Exception):
    """Raised for any structural problem that should map to ``BLOCK/hard_failure``."""


_RESULT_BADGE = {"PASS": "✅", "WARN": "⚠️", "BLOCK": "❌"}


def _emit(result: str, **fields: object) -> None:
    """Print the machine-parseable line and a markdown table to the CI summary.

    Args:
        result: One of ``PASS``, ``WARN``, ``BLOCK``.
        **fields: Additional ``key=value`` pairs (``kind=...`` for ``BLOCK``).
    """
    parts = [f"RESULT={result}"] + [f"{k}={v}" for k, v in fields.items()]
    print(" ".join(parts))
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            badge = _RESULT_BADGE.get(result, "")
            task = fields.get("task", "")
            rows = "\n".join(f"| {k} | {v} |" for k, v in fields.items() if k != "task")
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(f"### {badge} Perf Smoke: {result} — `{task}`\n\n| metric | value |\n|---|---|\n{rows}\n\n")
        except OSError:
            pass


# ----------------------------------------------------------------------------- IO


def _read_json(path: Path) -> object:
    """Read and parse a JSON file, returning whatever top-level value it holds."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise CompareError(f"file_not_found path={path}")
    except json.JSONDecodeError as e:
        raise CompareError(f"malformed_json path={path} line={e.lineno} col={e.colno}")


def _load_json(path: Path) -> dict:
    """Load a JSON file that must be a top-level object."""
    data = _read_json(path)
    if not isinstance(data, dict):
        raise CompareError(f"json_not_object path={path}")
    return data


def _normalize_result(data: object) -> dict:
    """Normalize a benchmark result into the OmniPerf ``{phase: {metric: value}}`` shape."""
    if isinstance(data, dict):
        return data
    if not isinstance(data, list):
        raise CompareError(f"json_not_object_or_list type={type(data).__name__}")
    out: dict[str, dict] = {}
    for phase in data:
        if not isinstance(phase, dict):
            continue
        name = phase.get("phase_name")
        if not isinstance(name, str) or not name:
            continue
        prefix = f"{_JSON_NAME_PREFIX} {name} "
        bucket: dict[str, object] = {}
        for entry in phase.get("measurements", []) or []:
            if isinstance(entry, dict) and "name" in entry and "value" in entry:
                key = str(entry["name"])
                bucket[key[len(prefix) :] if key.startswith(prefix) else key] = entry["value"]
        for entry in phase.get("metadata", []) or []:
            if isinstance(entry, dict) and "name" in entry and "data" in entry:
                key = str(entry["name"])
                bucket.setdefault(key[len(prefix) :] if key.startswith(prefix) else key, entry["data"])
        out[name] = bucket
    return out


def _load_result(path: Path) -> dict:
    """Load a benchmark result file and normalize it to the OmniPerf shape."""
    return _normalize_result(_read_json(path))


# ------------------------------------------------------------------- KPI extraction


def _runtime_arrays(result: dict) -> dict:
    """Return the ``Step Frametimes`` per-frame array map, or ``{}`` when absent."""
    runtime = result.get(METRIC_PHASE)
    if not isinstance(runtime, dict):
        return {}
    frametimes = runtime.get(FRAMETIMES_NAME)
    return frametimes if isinstance(frametimes, dict) else {}


def _floats(seq: object) -> list[float]:
    """Coerce a sequence into a list of plain floats (dropping bools/non-numerics)."""
    if not isinstance(seq, list):
        return []
    return [float(s) for s in seq if isinstance(s, (int, float)) and not isinstance(s, bool)]


def steady_fps(result: dict, warmup_frames: int, max_frames: int | None = None) -> float:
    """Mean post-warm-up effective FPS -- the gating KPI (D6).

    Computed as the mean of the per-frame ``Environment step effective FPS``
    array after dropping the first ``warmup_frames`` (and truncating to
    ``max_frames`` so a longer run is comparable to the calibrated window). Falls
    back to the scalar all-frames ``Mean Environment step effective FPS`` when the
    per-frame array is unavailable (e.g. the OmniPerf backend).

    Raises:
        CompareError: When neither the per-frame array nor the scalar metric
            yields a finite positive value.
    """
    arr = _floats(_runtime_arrays(result).get(EFF_FPS_ARRAY))
    if arr:
        window = arr[:max_frames] if max_frames else arr
        steady = window[warmup_frames:] if len(window) > warmup_frames else window
        if steady:
            value = sum(steady) / len(steady)
            if value == value and value > 0:
                return float(value)
    # Fallback: scalar all-frames metric.
    runtime = result.get(METRIC_PHASE)
    if isinstance(runtime, dict) and isinstance(runtime.get(METRIC_NAME), (int, float)):
        value = float(runtime[METRIC_NAME])
        if value == value and value > 0:
            return value
    raise CompareError(f"missing_metric phase={METRIC_PHASE} metric={METRIC_NAME!r}")


def _debug_kpis(result: dict, warmup_frames: int, max_frames: int | None = None) -> dict[str, object]:
    """Advisory per-frame KPIs for triage (never change the verdict).

    Reports the steady step-time distribution plus outlier accounting -- count,
    indices and magnitudes of steps slower than ``OUTLIER_FACTOR`` x the steady
    median (D-debug-info) -- and GPU memory when present.
    """
    arrays = _runtime_arrays(result)
    steps = _floats(arrays.get(STEP_MS_ARRAY))
    if not steps:
        return {}
    window = steps[:max_frames] if max_frames else steps
    if len(window) <= warmup_frames + 1:
        return {}
    steady = window[warmup_frames:]
    ordered = sorted(steady)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    mean = sum(steady) / n
    p99 = ordered[min(n - 1, int(round(0.99 * (n - 1))))]
    # Outlier indices are reported in the post-warm-up frame coordinate.
    outliers = [(i, s) for i, s in enumerate(steady) if median > 0 and s > OUTLIER_FACTOR * median]
    out: dict[str, object] = {
        "frames": len(window),
        "steady_median_ms": f"{median:.3f}",
        "steady_mean_ms": f"{mean:.3f}",
        "p99_over_median": f"{(p99 / median):.2f}" if median > 0 else "—",
        "outlier_count": len(outliers),
    }
    # Warm-up guardrail: the first *kept* frame should already look steady. If it is
    # much slower than the steady median, warmup_frames is likely too small and
    # one-time startup cost is leaking into the KPI. Advisory flag only -- the
    # verdict is unchanged; it tells a maintainer to re-check warmup_frames.
    if median > 0 and steady[0] > WARMUP_GUARD_FACTOR * median:
        out["warmup_flag"] = f"first_kept_frame={steady[0] / median:.1f}x_median(warmup={warmup_frames})"
    if outliers:
        idx = [i for i, _ in outliers[:MAX_REPORTED_OUTLIERS]]
        mag = [round(s / median, 2) for _, s in outliers[:MAX_REPORTED_OUTLIERS]]
        out["outlier_idx"] = ",".join(str(i) for i in idx)
        out["outlier_mag_x"] = ",".join(f"{m:g}" for m in mag)
    mem = result.get(METRIC_PHASE, {})
    if isinstance(mem, dict) and isinstance(mem.get("GPU Memory Used"), (int, float)):
        out["gpu_mem_gb"] = f"{float(mem['GPU Memory Used']):.2f}"
    return out


def _benchmark_info(result: dict) -> dict:
    """Return the run's self-reported config from the ``benchmark_info`` phase.

    The json backend records the ``task``/``seed``/``num_envs``/``num_frames`` the
    run actually used, plus the comma-joined ``presets`` (physics + renderer). Empty
    for the OmniPerf backend or older results.
    """
    info = result.get("benchmark_info")
    return info if isinstance(info, dict) else {}


def _expected_presets(task_entry: dict) -> list[str]:
    """Physics/renderer preset tokens the gate launches this task with.

    Pulled from the ``physics=`` / ``presets=`` Hydra overrides in ``benchmark_args``
    (e.g. ``physics=newton_mjwarp`` -> ``newton_mjwarp``). The backend echoes these
    back, comma-joined, in ``benchmark_info.presets``.
    """
    out: list[str] = []
    for arg in task_entry.get("benchmark_args", []) or []:
        if isinstance(arg, str) and "=" in arg:
            key, _, val = arg.partition("=")
            if key in ("physics", "presets") and val:
                out.append(val)
    return out


def _assert_run_config(result: dict, task: str, task_entry: dict) -> None:
    """Verify the run used the configured task settings; raise on drift (D1/provenance).

    The gate launches each task with the config carried in ``baseline.json``, and the
    backend echoes that config back in ``benchmark_info``. If the two disagree -- e.g.
    a PR changes a task's default ``num_envs`` -- the measured FPS is no longer
    comparable to the calibrated window, so a *config* change would be silently
    misread as a *perf* change. We treat that as a structural failure
    (``BLOCK/hard_failure``), not a regression. No-op when the backend reports no
    ``benchmark_info`` (OmniPerf / legacy results).
    """
    info = _benchmark_info(result)
    if not info:
        return
    want_task = task_entry.get("task_id", task)  # forward-compatible with task variants
    mismatches: list[str] = []
    ran_task = info.get("task")
    if isinstance(ran_task, str) and ran_task and ran_task != want_task:
        mismatches.append(f"task(ran={ran_task},want={want_task})")
    for field in ("num_envs", "seed"):
        want = task_entry.get(field)
        got = info.get(field)
        if want is not None and isinstance(got, (int, float)) and int(got) != int(want):
            mismatches.append(f"{field}(ran={int(got)},want={int(want)})")
    # The run must cover at least the calibrated frame count (the KPI truncates to it).
    want_frames = task_entry.get("num_frames")
    got_frames = info.get("num_frames")
    if want_frames is not None and isinstance(got_frames, (int, float)) and int(got_frames) < int(want_frames):
        mismatches.append(f"num_frames(ran={int(got_frames)},want>={int(want_frames)})")
    # Physics/renderer backend: every physics=/presets= override we launch with must
    # appear in the run's reported presets. Catches "ran PhysX when the @newton variant
    # was intended" -- a different KPI entirely, invisible to an FPS-only check.
    want_presets = _expected_presets(task_entry)
    if want_presets:
        ran_presets = {p.strip() for p in str(info.get("presets", "")).split(",") if p.strip()}
        if ran_presets:  # only assert when the backend reported its presets
            missing = [p for p in want_presets if p not in ran_presets]
            if missing:
                mismatches.append(f"presets(ran={','.join(sorted(ran_presets))},missing={','.join(missing)})")
    if mismatches:
        raise CompareError("config_mismatch " + " ".join(mismatches))


def _extract_provenance(result: dict) -> dict[str, object]:
    """Pull version provenance (warp / isaaclab / cuda); best-effort."""
    out: dict[str, object] = {}
    version = result.get("version_info")
    if isinstance(version, dict):
        for src, dst in (("warp_version", "warp"), ("isaaclab_version", "isaaclab")):
            val = version.get(src)
            if isinstance(val, str) and val:
                out[dst] = val
    hw = result.get("hardware_info")
    if isinstance(hw, dict):
        cuda = hw.get("cuda_version")
        if isinstance(cuda, str) and cuda:
            out["cuda"] = cuda
    return out


def _extract_gpu_name(result: dict) -> str | None:
    """Read the runner's GPU model name from the result's hardware metadata."""
    hw = result.get("hardware_info")
    if not isinstance(hw, dict):
        return None
    devices = hw.get("gpu_devices")
    if not isinstance(devices, dict) or not devices:
        return None
    current = str(hw.get("gpu_current_device", "0"))
    device = devices.get(current) or next(iter(devices.values()), None)
    if isinstance(device, dict):
        name = device.get("name")
        return name if isinstance(name, str) and name else None
    return None


# ------------------------------------------------------------------ robust stats


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    return ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def _median_mad(values: list[float]) -> tuple[float, float]:
    """Return ``(median, MAD)`` of ``values`` (MAD = median absolute deviation)."""
    center = _median(values)
    mad = _median([abs(v - center) for v in values])
    return center, mad


# ----------------------------------------------------------------- baseline / store


def _resolve_results(results_dir: str, glob_pattern: str, allow_multiple: bool) -> Path:
    """Resolve the result JSON path within ``results_dir`` (latest by sort order)."""
    matches = sorted(glob.glob(os.path.join(results_dir, glob_pattern)))
    if not matches:
        raise CompareError(f"no_results_found dir={results_dir!r} glob={glob_pattern!r}")
    if len(matches) > 1 and not allow_multiple:
        raise CompareError(f"multiple_results n={len(matches)}")
    return Path(matches[-1])


def _match_gpu(per_gpu: dict, gpu_key: str) -> tuple[str, dict]:
    """Find the baseline entry whose key (sub)string-matches ``gpu_key``."""
    for key, entry in per_gpu.items():
        if key == gpu_key or key in gpu_key or gpu_key in key:
            if not isinstance(entry, dict):
                raise CompareError(f"malformed_baseline_entry gpu={key!r}")
            return key, entry
    raise CompareError(f"baseline_gpu_mismatch gpu={gpu_key!r} known={sorted(per_gpu)}")


def _resolve_baseline(
    baseline: dict, task: str, gpu_name: str | None, gpu_override: str | None
) -> tuple[str, dict, dict]:
    """Return ``(gpu_key, task_entry, per_gpu_entry)`` for the task and GPU."""
    task_entry = baseline.get(task)
    if not isinstance(task_entry, dict):
        raise CompareError(f"missing_baseline_task task={task!r}")
    per_gpu = task_entry.get("per_gpu")
    if not isinstance(per_gpu, dict) or not per_gpu:
        raise CompareError(f"missing_per_gpu task={task!r}")
    gpu_key = gpu_override or gpu_name
    if not gpu_key:
        raise CompareError(f"unknown_gpu task={task!r}")
    matched_key, entry = _match_gpu(per_gpu, gpu_key)
    if "baseline_fps" not in entry:
        raise CompareError(f"missing_baseline_field task={task!r} gpu={matched_key!r} field=baseline_fps")
    return matched_key, task_entry, entry


def _history_window(history_dir: str | None, fingerprint: str | None, task: str, gpu_key: str) -> dict:
    """Load the rolling-window samples for ``(task, gpu)`` from the history store.

    The store mirrors the orphan branch: ``<history-dir>/<fingerprint>/<task>__<gpu>.json``
    with a flat ``<history-dir>/<task>__<gpu>.json`` fallback (the seeded
    "default" bucket). Returns ``{}`` when no window exists yet.
    """
    if not history_dir:
        return {}
    safe = f"{task}__{gpu_key}".replace("/", "_").replace(" ", "_")
    candidates = []
    if fingerprint:
        candidates.append(Path(history_dir) / fingerprint / f"{safe}.json")
    candidates.append(Path(history_dir) / f"{safe}.json")
    for path in candidates:
        if path.exists():
            data = _read_json(path)
            if isinstance(data, dict):
                return data
    return {}


def _safe_float(val: object) -> float | None:
    """Parse a float, returning ``None`` for missing / non-numeric values (e.g. ``"—"``)."""
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _window_values(window: dict, key: str) -> list[float]:
    """Pull a numeric series (``fps`` / ``wall_s``) out of a window's samples."""
    samples = window.get("samples")
    if not isinstance(samples, list):
        return []
    return [float(s[key]) for s in samples if isinstance(s, dict) and isinstance(s.get(key), (int, float))]


def _overrides_for(overrides: dict, task: str, gpu_key: str) -> dict:
    """Merge global defaults < per-task < per-task/gpu override blocks."""
    merged: dict[str, object] = {}
    defaults = overrides.get("_defaults")
    if isinstance(defaults, dict):
        merged.update(defaults)
    task_block = overrides.get(task)
    if isinstance(task_block, dict):
        merged.update({k: v for k, v in task_block.items() if not isinstance(v, dict)})
        gpu_block = task_block.get(gpu_key)
        if isinstance(gpu_block, dict):
            merged.update(gpu_block)
    return merged


def _thresholds(window: dict, entry: dict, ov: dict) -> tuple[float, float, float, float, str]:
    """Compute ``(center, spread, k_warn, k_block, source)`` for the FPS KPI.

    Uses the rolling-window median+MAD when enough samples exist; otherwise
    falls back to the static ``baseline_fps`` plus ``warn_pct`` / ``max_regression_pct``.
    Manual overrides (k values, spread floor, pinned center/spread) win.
    """
    k_warn = float(ov.get("k_warn", DEFAULT_K_WARN))
    k_block = float(ov.get("k_block", DEFAULT_K_BLOCK))
    min_spread_pct = float(ov.get("min_spread_pct", DEFAULT_MIN_SPREAD_PCT))

    fps_window = _window_values(window, "fps")
    if len(fps_window) >= MIN_WINDOW:
        center, mad = _median_mad(fps_window)
        spread = max(MAD_TO_STD * mad, min_spread_pct / 100.0 * center)
        source = f"window(n={len(fps_window)})"
    else:
        center = float(entry["baseline_fps"])
        # Map the static percent bands onto k-sigma so PASS/WARN/BLOCK math is uniform.
        warn_pct = float(entry.get("warn_pct", min_spread_pct * k_warn))
        block_pct = float(entry.get("max_regression_pct", min_spread_pct * k_block))
        spread = warn_pct / 100.0 * center / k_warn if k_warn else min_spread_pct / 100.0 * center
        # Honor the static block band exactly if it implies a different spread.
        spread = max(spread, block_pct / 100.0 * center / k_block if k_block else spread)
        source = "static_baseline"

    if "pin_center_fps" in ov:
        center = float(ov["pin_center_fps"])
        source = "override_pin"
    if "pin_spread_fps" in ov:
        spread = float(ov["pin_spread_fps"])
    return center, spread, k_warn, k_block, source


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--task",
        required=True,
        help="Gate key for baseline/history lookup, e.g. Isaac-Cartpole-v0 or Isaac-Cartpole-v0@newton.",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help="Gym task id used for the result glob and config check (defaults to --task; set for @variant keys).",
    )
    parser.add_argument("--results-dir", required=True, help="Directory containing the benchmark JSON.")
    parser.add_argument("--baseline", required=True, help="Path to baseline.json (run config + static fallback).")
    parser.add_argument("--history-dir", default=None, help="Rolling-window store (orphan-branch checkout).")
    parser.add_argument("--overrides", default=None, help="Path to baseline_overrides.json (committed with the PR).")
    parser.add_argument("--fingerprint", default=None, help="History bucket key (git-subtree+deps hash).")
    parser.add_argument("--measured-wall-s", type=float, default=None, help="Wall-clock seconds of the run.")
    parser.add_argument("--results-glob", default=None, help=f"Result glob (defaults to {DEFAULT_GLOB_TEMPLATE!r}).")
    parser.add_argument("--gpu-override", default=None, help="Override the GPU name read from the result JSON.")
    parser.add_argument("--allow-multiple", action="store_true", help="Permit multiple result files; pick the latest.")
    args = parser.parse_args(argv)

    # The gate key (--task) may be a variant like "Isaac-Cartpole-v0@newton"; the gym
    # task id (--task-id, defaulting to the part before "@") drives the file glob and
    # the config check, while the gate key drives baseline / history / overrides lookup.
    task_id = args.task_id or args.task.split("@", 1)[0]
    glob_pattern = args.results_glob or DEFAULT_GLOB_TEMPLATE.format(task=task_id)

    try:
        result_path = _resolve_results(args.results_dir, glob_pattern, args.allow_multiple)
        result = _load_result(result_path)
        baseline = _load_json(Path(args.baseline))
        gpu_name = _extract_gpu_name(result)
        gpu_key, task_entry, entry = _resolve_baseline(baseline, args.task, gpu_name, args.gpu_override)
        _assert_run_config(result, task_id, task_entry)
        warmup_frames = int(task_entry.get("warmup_frames", DEFAULT_WARMUP_FRAMES))
        max_frames = task_entry.get("num_frames")
        max_frames = int(max_frames) if isinstance(max_frames, (int, float)) else None
        measured_fps = steady_fps(result, warmup_frames, max_frames)
    except CompareError as e:
        _emit("BLOCK", kind="hard_failure", reason=str(e), task=args.task)
        return EXIT_HARD_FAILURE

    overrides = {}
    if args.overrides and Path(args.overrides).exists():
        overrides = _load_json(Path(args.overrides))
    ov = _overrides_for(overrides, args.task, gpu_key)

    if ov.get("skip"):
        _emit("PASS", task=args.task, gpu=gpu_key, note="skipped_by_override")
        return EXIT_PASS

    window = _history_window(args.history_dir, args.fingerprint, args.task, gpu_key)
    center, spread, k_warn, k_block, source = _thresholds(window, entry, ov)
    delta_pct = (measured_fps - center) / center * 100.0
    warn_floor = center - k_warn * spread
    block_floor = center - k_block * spread

    common: dict[str, object] = {
        "task": args.task,
        "gpu": gpu_key,
        "thresholds": source,
        "center_fps": f"{center:.0f}",
        "measured_fps": f"{measured_fps:.0f}",
        "delta_pct": f"{delta_pct:+.2f}",
        "warmup_frames": warmup_frames,
        "k_warn": f"{k_warn:g}",
        "k_block": f"{k_block:g}",
        "warn_below_fps": f"{warn_floor:.0f}",
        "block_below_fps": f"{block_floor:.0f}",
    }

    # Secondary signal (advisory): wall-clock vs the window's wall median.
    if args.measured_wall_s is not None:
        common["wall_s"] = f"{args.measured_wall_s:.0f}"
        wall_window = _window_values(window, "wall_s")
        if len(wall_window) >= MIN_WINDOW:
            wcenter, wmad = _median_mad(wall_window)
            wspread = max(MAD_TO_STD * wmad, DEFAULT_MIN_SPREAD_PCT / 100.0 * wcenter)
            common["wall_center_s"] = f"{wcenter:.0f}"
            common["wall_delta_pct"] = f"{(args.measured_wall_s - wcenter) / wcenter * 100.0:+.2f}"
            if args.measured_wall_s > wcenter + k_warn * wspread:
                common["wall_flag"] = "slow"

    common.update(_debug_kpis(result, warmup_frames, max_frames))
    common.update(_extract_provenance(result))

    # Advisory tail signal (opt-in): a per-task override can WARN when the post-warm-up
    # p99/median step-time ratio exceeds a ceiling -- a tail/spike regression the scalar
    # FPS mean hides. Off unless tail_p99_warn is set (tasks with real recurring spikes,
    # e.g. g1-flat first-ground-contact, would otherwise flag every run). Never blocks.
    tail_warn = False
    tail_ceiling = ov.get("tail_p99_warn")
    if tail_ceiling is not None:
        p99_ratio = _safe_float(common.get("p99_over_median"))
        if p99_ratio is not None and p99_ratio > float(tail_ceiling):
            common["tail_flag"] = f"p99_over_median={p99_ratio:g}>{float(tail_ceiling):g}"
            tail_warn = True

    if measured_fps < block_floor:
        _emit("BLOCK", kind="regression", **common)
        return EXIT_BLOCK
    if measured_fps < warn_floor:
        _emit("WARN", **common)
        return EXIT_PASS
    if tail_warn:
        _emit("WARN", reason="tail", **common)
        return EXIT_PASS
    _emit("PASS", **common)
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
