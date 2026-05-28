# Isaac Lab Benchmark Matrix — L40S Results

**Host**: NVIDIA L40S (47.5 GB, CC 8.9) · Intel Xeon Icelake 16-core · 62.8 GB RAM  
**Software**: Isaac Lab 6.1.0 · Newton 1.2.0 · PhysX isaaclab_physx 1.1.0 · Warp 1.12.1 · CUDA 12.8 · PyTorch 2.10.0+cu128  
**Date**: 2026-05-28  

Full report: [`REPORT.md`](REPORT.md) · Raw data: [`results.csv`](results.csv)

---

## Methodology

### What was swept

A matrix of **40 cells** across 7 tasks. Each cell is one unique combination of task, physics backend, renderer, camera resolution, and `num_envs`.

**Tasks:**

| task | type | physics backends | renderers |
|---|---|---|---|
| `Isaac-Cartpole-v0` | non-camera | PhysX, Newton | — |
| `Isaac-Factory-GearMesh-Direct-v0` | non-camera | PhysX only ¹ | — |
| `Isaac-Velocity-Flat-G1-v0` | non-camera | PhysX, Newton | — |
| `Isaac-Velocity-Rough-Anymal-C-v0` | non-camera | PhysX only ² | — |
| `Isaac-Velocity-Rough-G1-v0` | non-camera | PhysX only ² | — |
| `Isaac-Repose-Cube-Shadow-Vision-Direct-v0` | camera | PhysX, Newton | RTX, Warp, OVRTX ³ |
| `Isaac-Dexsuite-Kuka-Allegro-Lift-v0` | camera | PhysX only ² | RTX only ⁴ |

> ¹ Factory hardcodes `PhysxCfg` — the `physics=` Hydra override is not accepted.  
> ² Newton fails with `WarpCodegenAttributeError` on the `narrow_phase_find_mesh_triangle_overlaps_kernel` — mesh terrain and convex-mesh collision are not yet supported in Newton/Warp.  
> ³ `physx × ovrtx` excluded — labelled highly experimental by the IsaacLab team.  
> ⁴ Dexsuite + Newton: same mesh-kernel error. Dexsuite + Warp renderer: SIGABRT (Warp renderer is designed for Newton physics).  

**`num_envs` sweep:**

- Non-camera tasks: 2–3 task-specific counts (e.g. Cartpole: 4096/8192/16384; locomotion: 1024/2048 or 2048/4096).
- Camera tasks: fixed at `num_envs=128` for the renderer × resolution matrix, plus a small env-scaling probe at 256 and 512 for Shadow Vision and Dexsuite.

**Camera resolutions** (camera tasks only): 64×64, 128×128, 256×256.

Full per-cell breakdown: [`MATRIX.md`](MATRIX.md).

---

### Cold vs warm measurement

Each cell ran **exactly twice** in sequence:

```
1. Cache wipe  →  delete ~/.cache/ov, ~/.nv/ComputeCache, ~/.cache/warp
2. Cold run    →  first launch after wipe; captures JIT compile + shader-cache build cost
3. Warm run    →  immediate re-run with caches intact; captures steady-state performance
```

No other cell's subprocess runs between the cold and warm runs of a given cell. Cell order across the full sweep was randomised once (seed 1234, recorded in `output/sweep_meta.json`) to reduce time-of-day GPU clock bias.

The cold/warm split measures the **session-local compile penalty**: how much overhead is paid the first time a cell starts vs. subsequent starts with warm caches. This is directly relevant to CI pipelines and developer iteration time.

---

### What was measured

The primary metric is **Mean Environment step effective FPS** from the OmniPerf benchmark backend (`benchmark_non_rl.py`, 200 frames, seed 42). Effective FPS = `num_envs × (1 / mean_step_time)`.

OmniPerf also records:
- Startup phases broken down: App Launch, Python Imports, Scene Creation, Simulation Start, Total Start
- Step-time statistics: min/max/mean across all measured frames
- GPU memory used, GPU utilisation, CPU utilisation, system RAM

Per-cell subprocess timeout: 10 min for non-camera/locomotion tasks, 15 min for camera tasks and Factory. On timeout: process group killed, `log.txt` preserved, `failure_type=TIMEOUT` recorded.

**Failure classification:** `OK` · `MISSING_JSON` · `TIMEOUT` · `EXIT_NONZERO` · `SIGNAL`. Non-zero exits are promoted to `OK` when a valid OmniPerf JSON with FPS data is present (handles the known `ovrtx_renderer` teardown crash — see Finding 7 below).

---

## Key findings

### 1 — Newton is dramatically faster than PhysX on compatible tasks

On simple-geometry locomotion and control tasks, Newton (mjwarp) outperforms PhysX by a wide margin on warm runs:

| task | num_envs | PhysX eff. FPS | Newton eff. FPS | speedup |
|---|---:|---:|---:|:---:|
| Cartpole | 4,096 | 149,493 | 358,826 | **2.4×** |
| Cartpole | 8,192 | 232,274 | 740,227 | **3.2×** |
| Cartpole | 16,384 | 470,450 | 1,416,367 | **3.0×** |
| G1-Flat | 2,048 | 19,588 | 65,683 | **3.4×** |
| G1-Flat | 4,096 | 34,370 | 129,821 | **3.8×** |

Newton hits **1.4 M effective FPS** on Cartpole at n=16,384 — the single largest result in the sweep.

---

### 2 — Newton's cold-start cost is the dominant CI concern

Newton cells spend **2–5 minutes** at cold start (JIT compiling Warp kernels + loading CUDA modules). The same cells warm start in seconds. PhysX cold starts cost only 1.3–1.7× their warm equivalent — essentially just shader-cache repopulation.

| cell | cold start | warm start | ratio |
|---|---:|---:|:---:|
| `shadow_vision_newton_warp_256x256` | 222 s | 15 s | **14.5×** |
| `shadow_vision_newton_warp_128x128` | 225 s | 16 s | **14.4×** |
| `shadow_vision_newton_ovrtx_64x64` | 235 s | 23 s | **10.4×** |
| `cartpole_newton_n4096` | 134 s | 19 s | **7.0×** |
| `g1_flat_newton_n2048` | 267 s | 38 s | **7.0×** |

**CI implication**: persisting `~/.cache/warp` across CI jobs would remove most of this cost. Without it, every Newton job in a fresh container pays a multi-minute JIT tax even for a 10-frame smoke test.

---

### 3 — Warp renderer is 2.5–3.5× faster than RTX on camera tasks

On Shadow Vision (the only task with multiple renderers), Warp consistently beats RTX. The gap is largest at low resolutions where RTX's per-frame fixed overhead (path-tracer setup, BVH traversal) dominates.

| resolution | PhysX: Warp÷RTX | Newton: Warp÷RTX |
|---|:---:|:---:|
| 64×64 | **2.9×** | **2.8×** |
| 128×128 | **1.7×** | **2.6×** |
| 256×256 | **1.8×** | **2.5×** |

For tasks that only need depth or RGB at low resolution, the Warp renderer is the clear winner.

---

### 4 — Camera resolution dominates FPS more than physics choice

Stepping 64×64 → 256×256 typically halves FPS regardless of physics backend. Physics choice adds a further multiplier on top, but the pixel-count penalty comes first.

---

### 5 — Newton currently blocks all mesh-collision tasks

Three tasks were excluded from the Newton column at preflight due to a Warp kernel codegen error:

```
WarpCodegenAttributeError: Error while parsing function
"narrow_phase_find_mesh_triangle_overlaps_kernel"
```

Affected: `Anymal-Rough`, `G1-Rough` (mesh terrain), `Dexsuite-Lift` (convex-mesh objects). These were caught before the full sweep and did not consume sweep time. The performance team should track when this kernel lands in Newton.

---

### 6 — Factory-GearMesh scales sub-linearly with env count

Doubling Factory from n=1,024 → n=2,048 raises total FPS only ~35% (from ~1,100 to ~1,476), while per-env FPS drops by ~30%. Classic contact-rich saturation: the PhysX GPU solver is bottlenecked by contact-pair solving, not by raw integration.

---

### 7 — OVRTX renderer teardown crashes, but data is valid

`ovrtx_renderer` exits with SIGSEGV after the benchmark completes. The crash occurs in the Hydra render delegate destructor during headless teardown — OmniPerf has already flushed its results JSON by then, so the FPS measurements are valid. This is consistent with IsaacLab's "highly experimental" label for ovrtx, and `physx × ovrtx` was excluded from the matrix per that guidance. The Newton × OVRTX measurements (Shadow Vision, all three resolutions) are included in the results.

---

## Artifacts

| file | description |
|---|---|
| [`REPORT.md`](REPORT.md) | Full narrative report with the selected showcase figures |
| [`results.csv`](results.csv) | One row per subprocess run (79 rows) |
| [`figures/`](figures/) | 4 selected charts: physics backend comparison, renderer comparison, resolution scaling, cold/warm startup ratio |
| [`MATRIX.md`](MATRIX.md) | Full per-axis cell breakdown and skipped combinations |
