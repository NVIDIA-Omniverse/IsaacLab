#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-depth analysis of the json-backend matrix sweep.

Reads every ``<cell>/<run>`` directory under the sweep output, pulls per-run
metrics from the ``json`` backend result (mean effective FPS, per-frame step
times, startup-stage timings) plus the orchestration ``meta.json`` (wall time,
cache state, failure type), and derives the metrics the report is built on:

* cold-vs-warm steady-state FPS and startup time,
* first-frame pollution ratios (frame[0..2] / steady median),
* warm-to-warm variability (CV%, MAD, range/mean) and whether it shrinks with reps,
* tail metrics (exceedance rate, P99/median, excess kurtosis) via :mod:`tail_stats`,
* env-size / resolution scaling and a wall-clock timing budget.

Outputs ``results_full.csv`` (per run), ``cell_summary.csv`` (per cell), and a
printed findings summary. Figures are produced by ``make_figures.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR.parent / "tail_metric"))
import sweep_config as cfg  # noqa: E402
import tail_stats as ts  # noqa: E402

_COLD_DIR = "cold_round0"
_WARM_DIRS = [f"warm_round{i}" for i in range(1, cfg.WARM_RUNS + 1)]
_ALL_DIRS = [_COLD_DIR] + _WARM_DIRS

_STARTUP_KEYS = [
    "App Launch Time",
    "Python Imports Time",
    "Task Creation and Start Time",
    "Scene Creation Time",
    "Simulation Start Time",
    "Total Start Time (Launch to Train)",
]


def _result_json(run_dir: Path) -> Path | None:
    m = sorted(run_dir.glob("benchmark_non_rl_*.json"))
    return m[-1] if m else None


def _parse_phases(json_path: Path) -> dict:
    """Return {'fps', 'startup': {...}, 'step_times': [...]} from a json-backend file."""
    out: dict = {"fps": None, "startup": {}, "step_times": None}
    try:
        data = json.load(open(json_path))
    except Exception:
        return out
    if not isinstance(data, list):
        return out
    for ph in data:
        name = ph.get("phase_name")
        if name == "runtime":
            for m in ph.get("measurements", []):
                nm = str(m.get("name", ""))
                if nm.endswith("Mean Environment step effective FPS"):
                    out["fps"] = m.get("value")
                elif nm.endswith("Step Frametimes"):
                    times = (m.get("value") or {}).get("Environment step times")
                    if isinstance(times, list) and times:
                        out["step_times"] = times
        elif name == "startup":
            for m in ph.get("measurements", []):
                nm = str(m.get("name", ""))
                for key in _STARTUP_KEYS:
                    if nm.endswith(key):
                        out["startup"][key] = m.get("value")
    return out


def _load_run(run_dir: Path) -> dict | None:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.load(open(meta_path))
    except Exception:
        return None

    row: dict = {
        "cell": meta.get("label", run_dir.parent.name),
        "task": meta.get("task", ""),
        "physics": meta.get("physics", ""),
        "renderer": meta.get("renderer"),
        "resolution": tuple(meta["resolution"]) if meta.get("resolution") else None,
        "num_envs": meta.get("num_envs"),
        "cell_type": meta.get("cell_type", ""),
        "cache_state": meta.get("cache_state", ""),
        "run_index": meta.get("cell_run_index"),
        "failure_type": meta.get("failure_type", "MISSING_JSON"),
        "wall_sec": meta.get("wall_sec"),
        "has_camera": meta.get("renderer") is not None,
        "fps": None,
        "median_ms": None,
        "steady_median_ms": None,
        "frame0_ratio": None,
        "frame1_ratio": None,
        "frame2_ratio": None,
        "exceedance_rate": None,
        "p99_over_median": None,
        "excess_kurtosis": None,
        "total_start_ms": None,
        "n_frames": None,
    }
    if meta.get("failure_type") != "OK":
        return row

    jp = _result_json(run_dir)
    if jp is None:
        row["failure_type"] = "MISSING_JSON"
        return row
    p = _parse_phases(jp)
    row["fps"] = p["fps"]
    row["total_start_ms"] = p["startup"].get("Total Start Time (Launch to Train)")
    for k in _STARTUP_KEYS:
        row[f"startup::{k}"] = p["startup"].get(k)

    st = p["step_times"]
    if st:
        a = np.asarray(st, dtype=np.float64)
        row["n_frames"] = int(a.size)
        row["median_ms"] = float(np.median(a))
        try:
            tstat = ts.compute_tail_stats(a)
            row["steady_median_ms"] = tstat.steady_median_ms
            row["exceedance_rate"] = tstat.exceedance_rate
            row["p99_over_median"] = tstat.p99_over_median
            row["excess_kurtosis"] = tstat.excess_kurtosis
            sm = tstat.steady_median_ms
            if sm > 0:
                row["frame0_ratio"] = float(a[0] / sm)
                row["frame1_ratio"] = float(a[1] / sm) if a.size > 1 else None
                row["frame2_ratio"] = float(a[2] / sm) if a.size > 2 else None
        except ValueError:
            pass
    return row


def collect(output_root: Path) -> list[dict]:
    runs: list[dict] = []
    for cell_dir in sorted(output_root.iterdir()):
        if not cell_dir.is_dir() or cell_dir.name in {"preflight", "figures"}:
            continue
        for rn in _ALL_DIRS:
            rd = cell_dir / rn
            if rd.is_dir():
                row = _load_run(rd)
                if row is not None:
                    runs.append(row)
    return runs


def _cv_pct(vals: list[float]) -> float | None:
    if len(vals) < 2:
        return None
    a = np.asarray(vals, dtype=np.float64)
    m = float(a.mean())
    return float(a.std(ddof=1) / m * 100.0) if m else None


def _mad(vals: list[float]) -> float | None:
    if not vals:
        return None
    a = np.asarray(vals, dtype=np.float64)
    return float(np.median(np.abs(a - np.median(a))))


def summarize_cells(runs: list[dict]) -> list[dict]:
    from collections import defaultdict

    by_cell: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_cell[r["cell"]].append(r)

    out: list[dict] = []
    for cell, rows in sorted(by_cell.items()):
        ref = rows[0]
        cold = [r for r in rows if r["cache_state"] == "cold" and r["failure_type"] == "OK"]
        warm = [r for r in rows if r["cache_state"] == "warm" and r["failure_type"] == "OK"]
        warm_fps = [r["fps"] for r in warm if r["fps"]]
        cold_fps = [r["fps"] for r in cold if r["fps"]]
        n_ok = len([r for r in rows if r["failure_type"] == "OK"])

        def _med(key, src):
            v = [r[key] for r in src if r.get(key) is not None]
            return float(np.median(v)) if v else None

        warm_mean = float(np.mean(warm_fps)) if warm_fps else None
        out.append({
            "cell": cell,
            "task": ref["task"],
            "physics": ref["physics"],
            "renderer": ref["renderer"] or "-",
            "resolution": f"{ref['resolution'][0]}x{ref['resolution'][1]}" if ref["resolution"] else "-",
            "num_envs": ref["num_envs"],
            "cell_type": ref["cell_type"],
            "has_camera": ref["has_camera"],
            "n_ok_runs": n_ok,
            "cold_fps": float(np.mean(cold_fps)) if cold_fps else None,
            "warm_fps_mean": warm_mean,
            "warm_fps_cv_pct": _cv_pct(warm_fps),
            "warm_fps_mad": _mad(warm_fps),
            "warm_fps_range_over_mean": (max(warm_fps) - min(warm_fps)) / warm_mean if warm_fps and warm_mean else None,
            "warm_over_cold": (warm_mean / cold_fps[0] - 1.0) * 100 if warm_mean and cold_fps else None,
            "cold_startup_ms": _med("total_start_ms", cold),
            "warm_startup_ms": _med("total_start_ms", warm),
            "warm_steady_median_ms": _med("steady_median_ms", warm),
            "warm_exceedance_rate": _med("exceedance_rate", warm),
            "warm_p99_over_median": _med("p99_over_median", warm),
            "warm_excess_kurtosis": _med("excess_kurtosis", warm),
            "warm_frame0_ratio": _med("frame0_ratio", warm),
            "warm_frame1_ratio": _med("frame1_ratio", warm),
            "warm_frame2_ratio": _med("frame2_ratio", warm),
            "cold_wall_sec": _med("wall_sec", cold),
            "warm_wall_sec": _med("wall_sec", warm),
        })
    return out


def _write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(_SCRIPT_DIR / "output"))
    args = ap.parse_args()
    root = Path(args.output_dir)

    runs = collect(root)
    cells = summarize_cells(runs)

    run_fields = [
        "cell", "task", "physics", "renderer", "num_envs", "cell_type", "cache_state",
        "run_index", "failure_type", "fps", "wall_sec", "median_ms", "steady_median_ms",
        "frame0_ratio", "frame1_ratio", "frame2_ratio", "exceedance_rate", "p99_over_median",
        "excess_kurtosis", "total_start_ms", "n_frames",
    ]
    cell_fields = list(cells[0].keys()) if cells else []
    _write_csv(runs, root / "results_full.csv", run_fields)
    _write_csv(cells, root / "cell_summary.csv", cell_fields)
    print(f"wrote results_full.csv ({len(runs)} runs)  cell_summary.csv ({len(cells)} cells)")

    # ---- printed findings ----
    print("\n=== WARM VARIANCE: camera vs non-camera ===")
    cam = [c for c in cells if c["has_camera"] and c["warm_fps_cv_pct"] is not None]
    non = [c for c in cells if not c["has_camera"] and c["warm_fps_cv_pct"] is not None]
    for grp, name in ((non, "non-camera"), (cam, "camera")):
        cvs = [c["warm_fps_cv_pct"] for c in grp]
        if cvs:
            print(f"  {name:11s}: n={len(cvs):2d}  CV% median={np.median(cvs):.2f}  min={min(cvs):.2f}  max={max(cvs):.2f}")

    print("\n=== FIRST-FRAME POLLUTION (warm, median over cells) ===")
    for label, grp in (("non-camera", non), ("camera", cam)):
        f0 = [c["warm_frame0_ratio"] for c in grp if c["warm_frame0_ratio"]]
        f1 = [c["warm_frame1_ratio"] for c in grp if c["warm_frame1_ratio"]]
        f2 = [c["warm_frame2_ratio"] for c in grp if c["warm_frame2_ratio"]]
        if f0:
            print(f"  {label:11s}: frame0/med={np.median(f0):5.1f}x  frame1/med={np.median(f1):4.2f}x  frame2/med={np.median(f2):4.2f}x")

    print("\n=== COLD vs WARM (median over cells) ===")
    for label, grp in (("non-camera", non), ("camera", cam)):
        wc = [c["warm_over_cold"] for c in grp if c["warm_over_cold"] is not None]
        cs = [c["cold_startup_ms"] for c in grp if c["cold_startup_ms"]]
        ws = [c["warm_startup_ms"] for c in grp if c["warm_startup_ms"]]
        if wc:
            print(f"  {label:11s}: warm-vs-cold FPS={np.median(wc):+.1f}%  startup cold={np.median(cs)/1000:.1f}s warm={np.median(ws)/1000:.1f}s")

    print("\n=== TIMING BUDGET (warm wall sec, by cell) ===")
    tb = sorted([c for c in cells if c["warm_wall_sec"]], key=lambda c: c["warm_wall_sec"], reverse=True)
    for c in tb[:8]:
        print(f"  {c['cell']:42s} {c['warm_wall_sec']:6.1f}s")
    tot_warm = sum(c["warm_wall_sec"] for c in cells if c["warm_wall_sec"])
    print(f"  (sum of one warm run across all cells: {tot_warm/60:.1f} min)")

    print("\n=== TAIL: highest warm exceedance / kurtosis cells ===")
    tl = sorted([c for c in cells if c["warm_exceedance_rate"] is not None], key=lambda c: c["warm_exceedance_rate"], reverse=True)
    for c in tl[:8]:
        print(f"  {c['cell']:42s} exceed={c['warm_exceedance_rate']*100:5.2f}%  p99/med={c['warm_p99_over_median']:.2f}  kurt={c['warm_excess_kurtosis']:.1f}")


if __name__ == "__main__":
    main()
