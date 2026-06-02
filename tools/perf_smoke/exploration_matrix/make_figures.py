#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the report figures from the matrix-sweep output.

Produces a small, curated set of figures (not one per metric) under
``output/figures/``:

1. ``first_frame_pollution.png`` -- per-frame step-time traces for a camera and
   a non-camera cell, showing the frame[0..1] spike vs the steady median / 2x band.
2. ``cold_vs_warm_startup.png`` -- median Total Start Time, cold vs warm, per task.
3. ``warm_fps_variance.png`` -- warm FPS coefficient of variation per cell,
   colored by camera vs non-camera.
4. ``fps_scaling.png`` -- FPS vs num_envs (non-camera tasks) and FPS vs camera
   resolution (Shadow Vision).
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent / "tail_metric"))
import tail_stats as ts  # noqa: E402

OUT = _SCRIPT_DIR / "output"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

C_NON = "#2c7fb8"
C_CAM = "#d95f0e"

_TASK_SHORT = {
    "Isaac-Cartpole-v0": "Cartpole",
    "Isaac-Velocity-Flat-G1-v0": "G1-Flat",
    "Isaac-Velocity-Rough-G1-v0": "G1-Rough",
    "Isaac-Velocity-Rough-Anymal-C-v0": "Anymal-Rough",
    "Isaac-Factory-GearMesh-Direct-v0": "Factory",
    "Isaac-Repose-Cube-Shadow-Vision-Direct-v0": "Shadow-Vision",
    "Isaac-Dexsuite-Kuka-Allegro-Lift-v0": "Dexsuite-Lift",
}


def _short(task: str) -> str:
    return _TASK_SHORT.get(task, task.split("-")[1] if "-" in task else task)


def _read_cells() -> list[dict]:
    return list(csv.DictReader(open(OUT / "cell_summary.csv")))


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _trace(cell: str, run: str = "warm_round1") -> np.ndarray | None:
    d = OUT / cell / run
    try:
        return ts.load_trace(d)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Figure 1: first-frame pollution
# ---------------------------------------------------------------------------
def fig_first_frame(cells: list[dict]) -> None:
    examples = [
        ("g1_flat_physx_n2048", "Non-camera: G1-Flat (PhysX, 2048 envs)", C_NON),
        ("shadow_vision_physx_rtx_128x128_n128", "Camera: Shadow-Vision (PhysX+RTX, 128x128)", C_CAM),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (cell, title, color) in zip(axes, examples):
        tr = _trace(cell)
        if tr is None:
            continue
        n = min(45, len(tr))
        steady = float(np.median(tr[-50:]))
        ax.plot(range(n), tr[:n], "-o", ms=3, color=color, lw=1.2)
        ax.axhline(steady, color="#444", ls="--", lw=1, label=f"steady median = {steady:.2f} ms")
        ax.axhline(2 * steady, color="#999", ls=":", lw=1, label="2x median (filter cutoff)")
        ax.axvline(1.5, color="red", ls="-", lw=0.8, alpha=0.5)
        ax.text(1.7, ax.get_ylim()[1] * 0.92, "discard ->", color="red", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("frame index")
        ax.set_ylabel("step time [ms]")
        ax.legend(fontsize=8, loc="upper right")
        f0 = tr[0] / steady
        f1 = tr[1] / steady if len(tr) > 1 else float("nan")
        ax.annotate(
            f"frame0 = {f0:.1f}x\nframe1 = {f1:.1f}x",
            xy=(0.02, 0.62), xycoords="axes fraction", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec=color, alpha=0.8),
        )
        # Call out the worst mid-run hitch (>2x median, after the warmup window).
        body = tr[10:n]
        if body.size:
            j = int(np.argmax(body)) + 10
            if tr[j] > 2 * steady:
                ax.annotate(
                    f"mid-run hitch ({tr[j] / steady:.1f}x)\nsurvives any fixed discard",
                    xy=(j, tr[j]), xytext=(j - 18, tr[j] * 0.9), fontsize=8, color="red",
                    arrowprops=dict(arrowstyle="->", color="red", lw=1),
                )
    fig.suptitle(
        "Per-frame step times: frames 0-1 are JIT/alloc spikes (removed by drop-2), but mid-run hitches remain -> tail metric",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "first_frame_pollution.png", dpi=130)
    plt.close(fig)
    print("wrote first_frame_pollution.png")


# ---------------------------------------------------------------------------
# Figure 2: cold vs warm startup
# ---------------------------------------------------------------------------
def fig_startup(cells: list[dict]) -> None:
    by_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"cold": [], "warm": [], "cam": []})
    for c in cells:
        t = _short(c["task"])
        cold, warm = _f(c["cold_startup_ms"]), _f(c["warm_startup_ms"])
        if cold:
            by_task[t]["cold"].append(cold / 1000)
        if warm:
            by_task[t]["warm"].append(warm / 1000)
        by_task[t]["cam"].append(c["has_camera"] == "True")

    tasks = sorted(by_task, key=lambda t: -np.median(by_task[t]["cold"] or [0]))
    cold = [np.median(by_task[t]["cold"]) if by_task[t]["cold"] else 0 for t in tasks]
    warm = [np.median(by_task[t]["warm"]) if by_task[t]["warm"] else 0 for t in tasks]
    cam = [any(by_task[t]["cam"]) for t in tasks]

    x = np.arange(len(tasks))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.6))
    b1 = ax.bar(x - w / 2, cold, w, label="cold (caches wiped)", color="#bdbdbd", edgecolor="#555")
    b2 = ax.bar(x + w / 2, warm, w, label="warm (caches intact)", color="#31a354", edgecolor="#555")
    for i, (cd, wm) in enumerate(zip(cold, warm)):
        if cd and wm:
            ax.text(i, max(cd, wm) + 2, f"{cd/wm:.1f}x", ha="center", fontsize=8, color="#333")
    ax.set_xticks(x)
    labels = [f"{t}\n(camera)" if cam[i] else t for i, t in enumerate(tasks)]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Total Start Time (Launch -> Train) [s]")
    ax.set_title("Cold vs warm startup: shader/JIT caches amortize the cold-start cost (label = cold/warm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "cold_vs_warm_startup.png", dpi=130)
    plt.close(fig)
    print("wrote cold_vs_warm_startup.png")


# ---------------------------------------------------------------------------
# Figure 3: warm FPS variance, camera vs non-camera
# ---------------------------------------------------------------------------
def fig_variance(cells: list[dict]) -> None:
    rows = [(c["cell"], _f(c["warm_fps_cv_pct"]), c["has_camera"] == "True") for c in cells if _f(c["warm_fps_cv_pct"]) is not None]
    rows.sort(key=lambda r: r[1])
    labels = [r[0] for r in rows]
    cvs = [r[1] for r in rows]
    colors = [C_CAM if r[2] else C_NON for r in rows]

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(rows))
    ax.barh(y, cvs, color=colors, edgecolor="#444", lw=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("warm-run FPS coefficient of variation [%]  (lower = more reproducible)")
    ax.set_title("Run-to-run FPS variability is low for ALL tasks (<3.5%) and NOT higher for camera tasks")
    cam_med = np.median([r[1] for r in rows if r[2]])
    non_med = np.median([r[1] for r in rows if not r[2]])
    ax.axvline(non_med, color=C_NON, ls="--", lw=1.2, label=f"non-camera median = {non_med:.2f}%")
    ax.axvline(cam_med, color=C_CAM, ls="--", lw=1.2, label=f"camera median = {cam_med:.2f}%")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=C_NON, label="non-camera"),
        Patch(color=C_CAM, label="camera"),
        plt.Line2D([], [], color=C_NON, ls="--", label=f"non-cam median {non_med:.2f}%"),
        plt.Line2D([], [], color=C_CAM, ls="--", label=f"camera median {cam_med:.2f}%"),
    ], fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "warm_fps_variance.png", dpi=130)
    plt.close(fig)
    print("wrote warm_fps_variance.png")


# ---------------------------------------------------------------------------
# Figure 4: scaling
# ---------------------------------------------------------------------------
def fig_scaling(cells: list[dict]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    tasks = ["cartpole", "g1_flat", "g1_rough", "anymal_rough", "factory"]
    for task in tasks:
        pts = sorted(
            (int(c["num_envs"]), _f(c["warm_fps_mean"]))
            for c in cells
            if c["cell"].startswith(task) and c["physics"] == "physx" and _f(c["warm_fps_mean"])
        )
        if pts:
            xs, ys = zip(*pts)
            ax1.plot(xs, ys, "-o", label=_short(c["task"]) if False else task)
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("num_envs")
    ax1.set_ylabel("warm FPS (effective)")
    ax1.set_title("FPS scales ~linearly with num_envs (PhysX)")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", ls=":", alpha=0.4)

    res_order = {"64x64": 64, "128x128": 128, "256x256": 256}
    for phys, marker in (("physx", "o"), ("newton", "s")):
        pts = []
        for c in cells:
            if c["cell"].startswith(f"shadow_vision_{phys}_rtx") and c["num_envs"] == "128" and _f(c["warm_fps_mean"]):
                if c["resolution"] in res_order:
                    pts.append((res_order[c["resolution"]], _f(c["warm_fps_mean"])))
        pts.sort()
        if pts:
            xs, ys = zip(*pts)
            ax2.plot(xs, ys, f"-{marker}", label=f"Shadow-Vision {phys}+RTX")
    ax2.set_xlabel("camera resolution (px per side)")
    ax2.set_ylabel("warm FPS (effective)")
    ax2.set_title("Camera resolution: ~flat 64->128, modest cost at 256 (largely physics-bound)")
    ax2.set_xticks([64, 128, 256])
    ax2.legend(fontsize=8)
    ax2.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(FIG / "fps_scaling.png", dpi=130)
    plt.close(fig)
    print("wrote fps_scaling.png")


def main() -> None:
    cells = _read_cells()
    fig_first_frame(cells)
    fig_startup(cells)
    fig_variance(cells)
    fig_scaling(cells)
    print(f"\nfigures in {FIG}")


if __name__ == "__main__":
    main()
