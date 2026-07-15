# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Seed the rolling-window history store from existing L40S calibration runs.

The perf gate judges a run against a rolling window of historical samples
(median+MAD; see ``check_perf_regression.py``). On a fresh store there is no
window yet, so this one-off seeds it from the warm calibration runs already in
``exploration_matrix/output/`` -- the same L40S data the static baselines came
from -- computing each sample's post-warm-up steady FPS with the *identical*
function the gate uses, so the window and the gate agree by construction.

In production the window is the orphan branch and is appended to by
``rebaseline.py`` on the runner pool; this just gives the POC a realistic,
non-empty starting window. Run with any Python::

    python3 tools/perf_smoke/seed_history.py
"""

from __future__ import annotations

import contextlib
import csv
import datetime as dt
import glob
import json
import os
from pathlib import Path

from check_perf_regression import _load_result, env_fingerprint, history_basename, sample_provenance, steady_fps

_THIS_DIR = Path(__file__).resolve().parent
_MATRIX = _THIS_DIR / "exploration_matrix" / "output"
_RESULTS_CSV = _MATRIX / "results_full.csv"
_HISTORY_DIR = _THIS_DIR / "perf_history"
_GPU = "NVIDIA L40S"
_NUM_FRAMES = 300

# gate key -> (calibration cell, warmup_frames). The key may be a gym id (PhysX
# default) or a "<gym id>@<backend>" variant; the window file is named by the key.
_TASKS = {
    "Isaac-Cartpole-v0": ("cartpole_physx_n4096", 2),
    "Isaac-Factory-GearMesh-Direct-v0": ("factory_physx_n512", 2),
    "Isaac-Velocity-Flat-G1-v0": ("g1_flat_physx_n2048", 2),
    "Isaac-Repose-Cube-Shadow-Vision-Direct-v0": ("shadow_vision_physx_rtx_64x64_n128", 60),
    "Isaac-Cartpole-v0@newton": ("cartpole_newton_n4096", 5),
    "Isaac-Velocity-Flat-G1-v0@newton": ("g1_flat_newton_n2048", 5),
}


def _wall_by_round(cell: str) -> dict[int, float]:
    """Map warm run_index -> wall_sec for a cell from results_full.csv."""
    out: dict[int, float] = {}
    with open(_RESULTS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["cell"] == cell and row["cache_state"] == "warm" and row["failure_type"] == "OK":
                with contextlib.suppress(ValueError, KeyError):
                    out[int(row["run_index"])] = float(row["wall_sec"])
    return out


def main() -> int:
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for task, (cell, warmup) in _TASKS.items():
        walls = _wall_by_round(cell)
        samples = []
        fingerprint: str | None = None
        for rnd_dir in sorted(glob.glob(str(_MATRIX / cell / "warm_round*"))):
            run_index = int(os.path.basename(rnd_dir).replace("warm_round", ""))
            jsons = [p for p in sorted(glob.glob(os.path.join(rnd_dir, "*.json"))) if "meta" not in os.path.basename(p)]
            if not jsons:
                continue
            result = _load_result(Path(jsons[-1]))
            fps = steady_fps(result, warmup, _NUM_FRAMES)
            prov = sample_provenance(result)
            fingerprint = env_fingerprint(result) or fingerprint
            sample = {
                "fps": round(fps, 1),
                "wall_s": walls.get(run_index),
                "source": f"{cell}/{os.path.basename(rnd_dir)}",
                "ts": now,
            }
            sample.update({k: prov[k] for k in ("commit", "warp", "isaaclab", "cuda") if prov.get(k)})
            samples.append(sample)
        store = {
            "task": task,
            "gpu": _GPU,
            "num_frames": _NUM_FRAMES,
            "warmup_frames": warmup,
            "window": 20,
            "_note": "Seeded from L40S warm calibration runs (500f truncated to 300f, post-warm-up).",
            "samples": samples,
        }
        if fingerprint:
            store["fingerprint"] = fingerprint
        bucket = _HISTORY_DIR / fingerprint if fingerprint else _HISTORY_DIR
        bucket.mkdir(parents=True, exist_ok=True)
        out_path = bucket / f"{history_basename(task, _GPU)}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2)
            f.write("\n")
        fpss = [s["fps"] for s in samples]
        print(f"{task}: n={len(fpss)} fps={[round(x) for x in fpss]} -> {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
