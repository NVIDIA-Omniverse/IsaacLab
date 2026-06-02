# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Declarative matrix definition for the perf benchmark matrix sweep.

Import this module from run_sweep.py and analyze.py; do not run directly.
Edit the constants and TASK_CONFIGS list here to adjust the sweep without
touching the driver or analyzer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Backend / renderer constants
# ---------------------------------------------------------------------------

PHYSX = "physx"
NEWTON = "newton_mjwarp"

RTX = "isaacsim_rtx_renderer"
WARP = "newton_renderer"
OVRTX = "ovrtx_renderer"

# Human-readable short names used in cell labels and reports
PHYSICS_SHORT: dict[str, str] = {
    PHYSX: "physx",
    NEWTON: "newton",
}

RENDERER_SHORT: dict[str, str] = {
    RTX: "rtx",
    WARP: "warp",
    OVRTX: "ovrtx",
}

# ---------------------------------------------------------------------------
# Valid (physics, renderer) pairs for the camera matrix
# (PHYSX, OVRTX) is excluded — highly experimental per IsaacLab labelling
# ---------------------------------------------------------------------------

VALID_CAMERA_PAIRS: list[tuple[str, str]] = [
    (NEWTON, RTX),
    (NEWTON, WARP),
    (NEWTON, OVRTX),
    (PHYSX, RTX),
    (PHYSX, WARP),
    # (PHYSX, OVRTX) -- SKIPPED: highly experimental, in heavy development
]

# ---------------------------------------------------------------------------
# Camera resolution sweep
# ---------------------------------------------------------------------------

# Three resolution points swept in the camera matrix
CAMERA_RESOLUTIONS: list[tuple[int, int]] = [(64, 64), (128, 128), (256, 256)]

# Env-scaling probe: one fixed (renderer, resolution), two extra num_envs
ENV_SCALE_PROBE_RENDERER: str = RTX
ENV_SCALE_PROBE_RESOLUTION: tuple[int, int] = (128, 128)
ENV_SCALE_PROBE_NUM_ENVS: list[int] = [256, 512]

# ---------------------------------------------------------------------------
# Global sweep parameters
# ---------------------------------------------------------------------------

# Benchmark backend. "json" persists per-frame step times (needed to derive
# 100/300-frame stats offline, step-time shape, and outlier counts); "omniperf"
# writes aggregate stats only.
BACKEND: str = "json"

NUM_FRAMES: int = 500
SEED: int = 42
COLD_RUNS: int = 1
WARM_RUNS: int = 5
TOTAL_RUNS_PER_CELL: int = COLD_RUNS + WARM_RUNS

DEFAULT_TIMEOUT_SEC: int = 600    # 10 min  — non-camera / non-Factory tasks
LONG_TIMEOUT_SEC: int = 900       # 15 min  — camera tasks + Factory GearMesh
PREFLIGHT_TIMEOUT_SEC: int = 300  # 5 min   — preflight short runs
PREFLIGHT_NUM_FRAMES: int = 10

# ---------------------------------------------------------------------------
# Cell and task data structures
# ---------------------------------------------------------------------------


@dataclass
class TaskConfig:
    """Per-task configuration entry."""

    task_id: str
    """Registered gymnasium environment id."""

    label: str
    """Short identifier used in directory names and report tables."""

    physics_list: list[str]
    """Physics backends to sweep for this task."""

    num_envs_list: list[int]
    """
    Environment counts to sweep (non-camera tasks) or empty list
    (camera tasks use camera_anchor_num_envs for the matrix and
    ENV_SCALE_PROBE_NUM_ENVS for the probe).
    """

    has_camera: bool
    """Whether the task uses tiled cameras and needs --enable_cameras."""

    supports_physics_preset: bool = True
    """
    Whether the task registers a physics PresetCfg so the ``physics=<value>``
    Hydra override is accepted.  Tasks that hardcode their physics engine (e.g.
    Factory) set this to False; the builder will omit the override and run
    only one cell per num_envs rather than one per (num_envs, physics).
    """

    timeout_sec: int = 600
    """Per-subprocess wall-clock timeout in seconds."""

    camera_anchor_num_envs: int = 128
    """Fixed num_envs for the camera renderer x resolution matrix."""

    env_scale_probe: bool = False
    """Whether to run the env-scaling probe for this camera task."""

    # --- Resolution override strategy ---
    #
    # "hydra": pass ``env.tiled_camera.width=W env.tiled_camera.height=H``.
    #   Works for direct tasks whose env config has a plain tiled_camera field
    #   with width/height attributes accessible after PresetCfg resolution
    #   (e.g. Shadow Vision).
    #
    # "preset": resolution is a named preset variant on the camera PresetCfg;
    #   the correct preset name is looked up from resolution_preset_map.
    #   Works for tasks whose camera config uses resolution-named PresetCfg
    #   variants (e.g. Dexsuite's BaseTiledCameraCfg: rgb64, rgb128, rgb256).
    camera_resolution_mode: str = "hydra"

    resolution_width_key: str = "env.tiled_camera.width"
    resolution_height_key: str = "env.tiled_camera.height"

    resolution_preset_map: dict[tuple[int, int], str] = field(default_factory=dict)
    """
    Maps (W, H) tuples to preset names when ``camera_resolution_mode == "preset"``.
    Example for Dexsuite: {(64,64): "rgb64", (128,128): "rgb128", (256,256): "rgb256"}.
    """

    extra_presets: list[str] = field(default_factory=list)
    """
    Extra preset names always added to every cell for this task.
    Dexsuite needs ``["single_camera"]`` to wire the base_camera into the scene.
    These are emitted as ``presets=<name>`` tokens alongside physics and renderer.
    """

    renderer_extra_presets: dict[str, list[str]] = field(default_factory=dict)
    """
    Extra preset names added only when a specific renderer is selected.
    Shadow Vision needs ``{WARP: ["rgb"]}`` because the Warp renderer only
    supports ``depth`` and ``rgb`` data types, while the default Shadow Vision
    camera preset includes ``semantic_segmentation``.
    """

    excluded_renderers: list[str] = field(default_factory=list)
    """
    Renderers to skip for this task in the camera matrix.
    Dexsuite excludes the Warp renderer (crashes with SIGABRT when combined
    with PhysX; Warp renderer is primarily intended for Newton physics).
    """


@dataclass
class Cell:
    """One benchmark cell — a single (task, physics, renderer, resolution, num_envs) combination."""

    label: str
    """Unique slug used for directory names, e.g. cartpole_physx_n8192."""

    task: str
    physics: str

    renderer: str | None
    """Renderer preset name, or None for non-camera tasks."""

    resolution: tuple[int, int] | None
    """(W, H) in pixels, or None for non-camera tasks."""

    num_envs: int
    num_frames: int
    seed: int
    timeout_sec: int
    has_camera: bool
    cell_type: str
    """One of: non_camera | camera_matrix | camera_probe."""

    skip_physics_override: bool = False
    """
    When True, omit the ``physics=<value>`` Hydra arg because the task does not
    register a physics PresetCfg (e.g. Factory).  The ``physics`` field is still
    used as a metadata label.
    """

    extra_presets: list[str] = field(default_factory=list)
    """
    Extra ``presets=NAME`` tokens appended after physics and renderer.
    Used for Dexsuite's scene-level preset (e.g. ``["single_camera"]``).
    """

    resolution_preset: str | None = None
    """
    When set, emit ``presets=<resolution_preset>`` instead of width/height Hydra
    overrides.  Used for Dexsuite's ``BaseTiledCameraCfg`` named presets
    (``rgb64``, ``rgb128``, ``rgb256``).
    """

    resolution_width_key: str = "env.tiled_camera.width"
    resolution_height_key: str = "env.tiled_camera.height"

    def cli_args(self) -> list[str]:
        """Return the Hydra-style positional args to append to the benchmark command."""
        args: list[str] = []
        # Physics override — skipped for tasks that don't register a physics PresetCfg
        if not self.skip_physics_override:
            args.append(f"physics={self.physics}")
        # Renderer preset (camera tasks only)
        if self.renderer is not None:
            args.append(f"presets={self.renderer}")
        # Extra scene-level presets (e.g. Dexsuite "single_camera")
        for p in self.extra_presets:
            args.append(f"presets={p}")
        # Resolution: either a named preset or direct width/height Hydra overrides
        if self.resolution is not None:
            if self.resolution_preset is not None:
                args.append(f"presets={self.resolution_preset}")
            else:
                w, h = self.resolution
                args.append(f"{self.resolution_width_key}={w}")
                args.append(f"{self.resolution_height_key}={h}")
        return args


# ---------------------------------------------------------------------------
# Task configurations
# ---------------------------------------------------------------------------

TASK_CONFIGS: list[TaskConfig] = [
    TaskConfig(
        task_id="Isaac-Cartpole-v0",
        label="cartpole",
        physics_list=[PHYSX, NEWTON],
        num_envs_list=[4096, 8192, 16384],
        has_camera=False,
        timeout_sec=DEFAULT_TIMEOUT_SEC,
    ),
    TaskConfig(
        task_id="Isaac-Factory-GearMesh-Direct-v0",
        label="factory",
        # Factory hardcodes PhysxCfg directly (no PresetCfg) so the physics=
        # Hydra override is not accepted.  We record "physx" as a label but do
        # not pass the override.  physics_list has one entry → one cell per
        # num_envs instead of two.
        physics_list=[PHYSX],
        supports_physics_preset=False,
        # small/medium/large. 4096 excluded: known PhysX GPU collision-stack buffer overflow.
        num_envs_list=[512, 1024, 2048],
        has_camera=False,
        timeout_sec=LONG_TIMEOUT_SEC,
    ),
    TaskConfig(
        task_id="Isaac-Velocity-Flat-G1-v0",
        label="g1_flat",
        physics_list=[PHYSX, NEWTON],
        # small/medium/large
        num_envs_list=[1024, 2048, 4096],
        has_camera=False,
        timeout_sec=DEFAULT_TIMEOUT_SEC,
    ),
    TaskConfig(
        task_id="Isaac-Velocity-Rough-Anymal-C-v0",
        label="anymal_rough",
        # Newton fails with WarpCodegenAttributeError on mesh terrain collision
        # (narrow_phase_find_mesh_triangle_overlaps_kernel).
        physics_list=[PHYSX],
        # small/medium/large
        num_envs_list=[512, 1024, 2048],
        has_camera=False,
        timeout_sec=DEFAULT_TIMEOUT_SEC,
    ),
    TaskConfig(
        task_id="Isaac-Velocity-Rough-G1-v0",
        label="g1_rough",
        # Newton fails with WarpCodegenAttributeError on mesh terrain collision.
        physics_list=[PHYSX],
        # small/medium/large
        num_envs_list=[512, 1024, 2048],
        has_camera=False,
        timeout_sec=DEFAULT_TIMEOUT_SEC,
    ),
    TaskConfig(
        task_id="Isaac-Repose-Cube-Shadow-Vision-Direct-v0",
        label="shadow_vision",
        physics_list=[PHYSX, NEWTON],
        num_envs_list=[],  # camera task — uses camera_anchor_num_envs
        has_camera=True,
        timeout_sec=LONG_TIMEOUT_SEC,
        camera_anchor_num_envs=128,
        env_scale_probe=True,
        # ShadowHandVisionTiledCameraCfg is a PresetCfg but Hydra resolves it
        # to _ShadowHandBaseTiledCameraCfg (the 'default' variant) before
        # applying overrides, so the path is the same flat key as dexsuite.
        resolution_width_key="env.tiled_camera.width",
        resolution_height_key="env.tiled_camera.height",
        # The Warp renderer only supports ['depth', 'rgb']; Shadow Vision's
        # default camera preset includes semantic_segmentation, which triggers
        # a ValueError at launch.  Adding 'rgb' selects the rgb-only variant.
        renderer_extra_presets={WARP: ["rgb"]},
    ),
    TaskConfig(
        task_id="Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
        label="dexsuite_lift",
        # Newton fails with WarpCodegenAttributeError on mesh collision geometry
        # (same kernel as rough terrain: narrow_phase_find_mesh_triangle_overlaps).
        physics_list=[PHYSX],
        num_envs_list=[],  # camera task — uses camera_anchor_num_envs
        has_camera=True,
        timeout_sec=LONG_TIMEOUT_SEC,
        camera_anchor_num_envs=128,
        env_scale_probe=True,
        # Dexsuite uses scene-level camera presets, not a top-level tiled_camera
        # field.  "single_camera" wires BaseTiledCameraCfg() into the scene and
        # the resolution is selected via BaseTiledCameraCfg's named variants.
        camera_resolution_mode="preset",
        extra_presets=["single_camera"],
        resolution_preset_map={(64, 64): "rgb64", (128, 128): "rgb128", (256, 256): "rgb256"},
        # Warp renderer crashes with SIGABRT when paired with PhysX on this host.
        # Warp renderer is designed for Newton physics; skip it for Dexsuite
        # which only runs with PhysX.
        excluded_renderers=[WARP],
    ),
]


# ---------------------------------------------------------------------------
# Cell builders
# ---------------------------------------------------------------------------


def _make_label(task_label: str, physics: str, renderer: str | None, resolution: tuple[int, int] | None, num_envs: int) -> str:
    """Build a short unique slug for a cell."""
    parts = [task_label, PHYSICS_SHORT[physics]]
    if renderer is not None:
        parts.append(RENDERER_SHORT[renderer])
    if resolution is not None:
        w, h = resolution
        parts.append(f"{w}x{h}")
    parts.append(f"n{num_envs}")
    return "_".join(parts)


def build_cells() -> list[Cell]:
    """Return the full list of cells for the sweep.

    Three groups are generated in order (the driver shuffles them):
    1. Non-camera physics sweep (Part 1 in MATRIX.md)
    2. Camera renderer x resolution matrix (Part 2)
    3. Camera env-scaling probe (Part 3)
    """
    cells: list[Cell] = []

    for tc in TASK_CONFIGS:
        if not tc.has_camera:
            # --- Part 1: non-camera physics sweep ---
            for physics in tc.physics_list:
                    for num_envs in tc.num_envs_list:
                        cells.append(
                            Cell(
                                label=_make_label(tc.label, physics, None, None, num_envs),
                                task=tc.task_id,
                                physics=physics,
                                renderer=None,
                                resolution=None,
                                num_envs=num_envs,
                                num_frames=NUM_FRAMES,
                                seed=SEED,
                                timeout_sec=tc.timeout_sec,
                                has_camera=False,
                                cell_type="non_camera",
                                skip_physics_override=not tc.supports_physics_preset,
                            )
                        )
        else:
            # --- Part 2: camera renderer x resolution matrix ---
            for physics, renderer in VALID_CAMERA_PAIRS:
                if physics not in tc.physics_list:
                    continue
                if renderer in tc.excluded_renderers:
                    continue
                for resolution in CAMERA_RESOLUTIONS:
                    res_preset = tc.resolution_preset_map.get(resolution) if tc.camera_resolution_mode == "preset" else None
                    rend_extra = tc.renderer_extra_presets.get(renderer, [])
                    cells.append(
                        Cell(
                            label=_make_label(tc.label, physics, renderer, resolution, tc.camera_anchor_num_envs),
                            task=tc.task_id,
                            physics=physics,
                            renderer=renderer,
                            resolution=resolution,
                            num_envs=tc.camera_anchor_num_envs,
                            num_frames=NUM_FRAMES,
                            seed=SEED,
                            timeout_sec=tc.timeout_sec,
                            has_camera=True,
                            cell_type="camera_matrix",
                            extra_presets=list(tc.extra_presets) + list(rend_extra),
                            resolution_preset=res_preset,
                            resolution_width_key=tc.resolution_width_key,
                            resolution_height_key=tc.resolution_height_key,
                        )
                    )

            # --- Part 3: camera env-scaling probe ---
            if tc.env_scale_probe:
                for physics in tc.physics_list:
                    for num_envs in ENV_SCALE_PROBE_NUM_ENVS:
                        probe_res_preset = (
                            tc.resolution_preset_map.get(ENV_SCALE_PROBE_RESOLUTION)
                            if tc.camera_resolution_mode == "preset"
                            else None
                        )
                        probe_rend_extra = tc.renderer_extra_presets.get(ENV_SCALE_PROBE_RENDERER, [])
                        cells.append(
                            Cell(
                                label=_make_label(
                                    tc.label,
                                    physics,
                                    ENV_SCALE_PROBE_RENDERER,
                                    ENV_SCALE_PROBE_RESOLUTION,
                                    num_envs,
                                ),
                                task=tc.task_id,
                                physics=physics,
                                renderer=ENV_SCALE_PROBE_RENDERER,
                                resolution=ENV_SCALE_PROBE_RESOLUTION,
                                num_envs=num_envs,
                                num_frames=NUM_FRAMES,
                                seed=SEED,
                                timeout_sec=tc.timeout_sec,
                                has_camera=True,
                                cell_type="camera_probe",
                                extra_presets=list(tc.extra_presets) + list(probe_rend_extra),
                                resolution_preset=probe_res_preset,
                                resolution_width_key=tc.resolution_width_key,
                                resolution_height_key=tc.resolution_height_key,
                            )
                        )

    return cells


def build_preflight_cells() -> list[Cell]:
    """Return one short cell per unique (task, physics, renderer) compatibility key.

    Resolution is not part of the preflight key — one width is enough to
    verify the Hydra override path.  Cold/warm is not tested; preflight
    always runs without cache management.
    """
    seen: set[tuple[str, str, str | None]] = set()
    cells: list[Cell] = []

    for tc in TASK_CONFIGS:
        if not tc.has_camera:
               for physics in tc.physics_list:
                   key = (tc.task_id, physics, None)
                   if key in seen:
                       continue
                   seen.add(key)
                   num_envs = tc.num_envs_list[0] if tc.num_envs_list else tc.camera_anchor_num_envs
                   cells.append(
                       Cell(
                           label=f"pf_{_make_label(tc.label, physics, None, None, num_envs)}",
                           task=tc.task_id,
                           physics=physics,
                           renderer=None,
                           resolution=None,
                           num_envs=num_envs,
                           num_frames=PREFLIGHT_NUM_FRAMES,
                           seed=SEED,
                           timeout_sec=PREFLIGHT_TIMEOUT_SEC,
                           has_camera=False,
                           cell_type="non_camera",
                           skip_physics_override=not tc.supports_physics_preset,
                       )
                   )
        else:
            for physics, renderer in VALID_CAMERA_PAIRS:
                if physics not in tc.physics_list:
                    continue
                if renderer in tc.excluded_renderers:
                    continue
                key = (tc.task_id, physics, renderer)
                if key in seen:
                    continue
                seen.add(key)
                # Use the middle resolution for the preflight so the override
                # path is exercised without being the trivial minimum
                resolution = CAMERA_RESOLUTIONS[1]  # 128x128
                pf_res_preset = (
                    tc.resolution_preset_map.get(resolution)
                    if tc.camera_resolution_mode == "preset"
                    else None
                )
                pf_rend_extra = tc.renderer_extra_presets.get(renderer, [])
                cells.append(
                    Cell(
                        label=f"pf_{_make_label(tc.label, physics, renderer, resolution, tc.camera_anchor_num_envs)}",
                        task=tc.task_id,
                        physics=physics,
                        renderer=renderer,
                        resolution=resolution,
                        num_envs=tc.camera_anchor_num_envs,
                        num_frames=PREFLIGHT_NUM_FRAMES,
                        seed=SEED,
                        timeout_sec=PREFLIGHT_TIMEOUT_SEC,
                        has_camera=True,
                        cell_type="camera_matrix",
                        extra_presets=list(tc.extra_presets) + list(pf_rend_extra),
                        resolution_preset=pf_res_preset,
                        resolution_width_key=tc.resolution_width_key,
                        resolution_height_key=tc.resolution_height_key,
                    )
                )

    return cells
