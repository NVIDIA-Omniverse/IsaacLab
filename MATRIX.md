# Benchmark Matrix — exploration_matrix sweep

Sweep across 7 tasks varying physics backend, renderer (camera tasks only),
camera resolution (camera tasks only), `num_envs`, and cold vs warm.
Each cell is run 2 times: 1 cold (caches wiped) + 1 warm (caches preserved).

---

## Part 1 — Non-camera physics sweep (16 cells)

Each cell = `(task, num_envs, physics)`. No renderer, no resolution.

| task | num_envs sweep | physx | newton_mjwarp | cells |
|---|---|:---:|:---:|:---:|
| `Isaac-Cartpole-v0` | 4096, 8192, 16384 | 3 | 3 | **6** |
| `Isaac-Factory-GearMesh-Direct-v0` | 1024, 2048 | 2 | — | **2** |
| `Isaac-Velocity-Flat-G1-v0` | 2048, 4096 | 2 | 2 | **4** |
| `Isaac-Velocity-Rough-Anymal-C-v0` | 1024, 2048 | 2 | — | **2** |
| `Isaac-Velocity-Rough-G1-v0` | 1024, 2048 | 2 | — | **2** |
| | | | **subtotal** | **16** |

> Factory does not register a `physics` `PresetCfg` (hardcodes `PhysxCfg`),
> so the `physics=` Hydra override is omitted and Newton cells are skipped.
> Factory `num_envs=4096` excluded: triggers PhysX GPU collision-stack overflow.
>
> Rough terrain tasks (Anymal-C, G1-Rough) are PhysX-only: Newton fails with
> `WarpCodegenAttributeError` in `narrow_phase_find_mesh_triangle_overlaps_kernel`
> (mesh terrain collision not yet supported in Newton/Warp).

---

## Part 2 — Camera renderer × resolution matrix (18 cells)

Fixed `num_envs=128` for both camera tasks.

| physics | renderer | 64×64 | 128×128 | 256×256 | cells per row |
|---|---|:---:|:---:|:---:|:---:|
| `newton_mjwarp` | `isaacsim_rtx_renderer` | 1 | 1 | 1 | 3 |
| `newton_mjwarp` | `newton_renderer` (Warp) + `rgb` | 1 | 1 | 1 | 3 |
| `newton_mjwarp` | `ovrtx_renderer` | 1 | 1 | 1 | 3 |
| `physx` | `isaacsim_rtx_renderer` | 1 | 1 | 1 | 3 |
| `physx` | `newton_renderer` (Warp) + `rgb` | 1 | 1 | 1 | 3 |

**Shadow Vision** (`Isaac-Repose-Cube-Shadow-Vision-Direct-v0`): all 5 × 3 = **15 cells**.
Uses `env.tiled_camera.width=W env.tiled_camera.height=H` override.

**Dexsuite** (`Isaac-Dexsuite-Kuka-Allegro-Lift-v0`): PhysX + RTX only, 1 × 3 = **3 cells**.
Uses `presets=single_camera,rgb64/rgb128/rgb256`.

**Total Part 2: 18 cells.**

> Excluded combinations and reasons:
> - `physx × ovrtx_renderer`: highly experimental per IsaacLab team; skipped from the matrix.
> - Dexsuite + Newton: same Warp mesh-collision kernel error as rough terrain.
> - Dexsuite + Warp renderer: crashes with SIGABRT (Warp renderer is designed
>   for Newton physics; running it with PhysX on Dexsuite is unsupported).
> - Shadow Vision + Warp renderer: default preset includes `semantic_segmentation`
>   which Warp does not support; fixed by adding `presets=rgb`.
> - `ovrtx_renderer` exits with SIGSEGV on teardown in headless mode, but the
>   benchmark JSON is written before the crash. Results are valid; the driver
>   promotes such runs from `EXIT_NONZERO` to `OK` when a valid FPS metric is
>   present in the JSON.

---

## Part 3 — Camera env-scaling probe (6 cells)

One representative `(renderer=isaacsim_rtx_renderer, resolution=128×128)` point
scaled over two additional `num_envs` values.

| task | physics | renderer | resolution | num_envs | cells |
|---|---|---|---|---|:---:|
| `Shadow-Vision-Direct` | `physx` | `isaacsim_rtx_renderer` | 128×128 | 256, 512 | 2 |
| `Shadow-Vision-Direct` | `newton_mjwarp` | `isaacsim_rtx_renderer` | 128×128 | 256, 512 | 2 |
| `Dexsuite-Kuka-Allegro-Lift` | `physx` | `isaacsim_rtx_renderer` | 128×128 | 256, 512 | 2 |
| | | | | **subtotal** | **6** |

---

## Totals

| group | cells | repeats per cell | subprocess runs |
|---|:---:|:---:|:---:|
| Non-camera physics sweep | 16 | 2 (1c + 1w) | 32 |
| Camera renderer × resolution matrix | 18 | 2 (1c + 1w) | 36 |
| Camera env-scaling probe | 6 | 2 (1c + 1w) | 12 |
| **Full sweep** | **40** | | **80** |
| Preflight (separate, always first) | 13 | 1 | 13 |

> Actual measured runs: 79. One cell (`dexsuite_lift_physx_rtx_256x256_n128`) had its
> cold run fail with `EXIT_NONZERO` before producing an OmniPerf JSON, and its
> warm run was not attempted under the all-or-nothing cell policy. Final sweep
> success rate: **39 / 40 cells**.

---

## Valid (physics, renderer) combinations by task

| task | newton × rtx | newton × warp | newton × ovrtx | physx × rtx | physx × warp |
|---|:---:|:---:|:---:|:---:|:---:|
| Shadow Vision | yes | yes (rgb only) | yes | yes | yes (rgb only) |
| Dexsuite | — | — | — | yes | — |

---

## Per-task timeout budget

| task family | timeout per subprocess |
|---|---|
| Non-camera (Cartpole, locomotion) | 10 min |
| Factory GearMesh | 15 min |
| Camera tasks (Shadow Vision, Dexsuite-Lift) | 15 min |

On timeout: process group is killed (SIGTERM then SIGKILL), `log.txt` is
preserved, `failure_type=TIMEOUT` is recorded in `meta.json`.

---

## Execution notes

- Cell order is randomized once via a seeded shuffle (seed 1234 recorded in
  `output/sweep_meta.json`) to reduce time-of-day bias across cells.
- Within each cell, subprocesses always run in order:
  cache-wipe → cold → warm #1.
  No other cell's subprocess runs in between.
