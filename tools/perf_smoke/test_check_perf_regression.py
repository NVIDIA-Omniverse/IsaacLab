# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ``check_perf_regression.py`` (the perf-smoke comparator).

Covers the test logic of ``ci-regression-gate-config-info.md``: the post-warm-up
steady metric (D6), rolling-window median+MAD thresholds with a static fallback,
the ``PASS/WARN/BLOCK`` vocabulary, the advisory wall-clock signal, outlier
index/magnitude reporting, and manual overrides.

Stdlib ``unittest``; also collectable by pytest via this directory's ``pytest.ini``.
Run directly::

    python3 tools/perf_smoke/test_check_perf_regression.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_perf_regression as cpr  # noqa: E402

TASK = "Isaac-Cartpole-Direct-v0"
GPU = "NVIDIA L40"
BASELINE_FPS = 100000


def _omni_doc(fps: float | int | str | None, gpu: str | None = GPU) -> dict:
    """OmniPerf-shaped result (scalar metric, no per-frame array)."""
    doc: dict = {"runtime": {cpr.METRIC_NAME: fps}}
    if gpu is not None:
        doc["hardware_info"] = {"gpu_current_device": 0, "gpu_devices": {"0": {"name": gpu}}}
    return doc


def _eff_doc(
    eff_fps: list[float] | None,
    gpu: str = GPU,
    step_times: list[float] | None = None,
    scalar: float | None = None,
) -> list:
    """json-backend (list-of-phases) result carrying the per-frame arrays."""
    ft: dict = {}
    if eff_fps is not None:
        ft[cpr.EFF_FPS_ARRAY] = eff_fps
    if step_times is not None:
        ft[cpr.STEP_MS_ARRAY] = step_times
    runtime_meas: list = []
    if scalar is not None:
        runtime_meas.append({"name": f"benchmark_non_rl runtime {cpr.METRIC_NAME}", "value": scalar})
    if ft:
        runtime_meas.append({"name": f"benchmark_non_rl runtime {cpr.FRAMETIMES_NAME}", "value": ft})
    return [
        {"phase_name": "runtime", "measurements": runtime_meas, "metadata": []},
        {
            "phase_name": "hardware_info",
            "measurements": [],
            "metadata": [
                {"name": "benchmark_non_rl hardware_info gpu_current_device", "data": 0},
                {"name": "benchmark_non_rl hardware_info gpu_devices", "data": {"0": {"name": gpu}}},
            ],
        },
    ]


def _baseline(
    fps: int = BASELINE_FPS,
    warn: float = 5.0,
    block: float = 10.0,
    gpu_key: str = GPU,
    warmup: int | None = None,
    num_frames: int | None = None,
) -> dict:
    """Minimal baseline document (run config + static fallback thresholds)."""
    task_entry: dict = {"per_gpu": {gpu_key: {"baseline_fps": fps, "warn_pct": warn, "max_regression_pct": block}}}
    if warmup is not None:
        task_entry["warmup_frames"] = warmup
    if num_frames is not None:
        task_entry["num_frames"] = num_frames
    return {TASK: task_entry}


class _GateTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.results_dir = self.tmp / "perf-output"
        self.results_dir.mkdir()
        self.baseline_path = self.tmp / "baseline.json"
        self.history_dir = self.tmp / "history"
        self.overrides_path = self.tmp / "overrides.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_result(self, doc: dict | list | str, name: str | None = None) -> None:
        path = self.results_dir / (name or f"benchmark_non_rl_{TASK}_x.json")
        path.write_text(doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8")

    def _write_baseline(self, doc: dict) -> None:
        self.baseline_path.write_text(json.dumps(doc), encoding="utf-8")

    def _write_window(self, fps: list[float], wall: list[float] | None = None, gpu_key: str = GPU) -> None:
        self.history_dir.mkdir(exist_ok=True)
        samples = [{"fps": v} for v in fps]
        if wall is not None:
            for s, w in zip(samples, wall):
                s["wall_s"] = w
        safe = f"{TASK}__{gpu_key}".replace(" ", "_")
        (self.history_dir / f"{safe}.json").write_text(json.dumps({"samples": samples}), encoding="utf-8")

    def _write_overrides(self, doc: dict) -> None:
        self.overrides_path.write_text(json.dumps(doc), encoding="utf-8")

    def _run(self, *extra: str) -> tuple[int, str]:
        argv = ["--task", TASK, "--results-dir", str(self.results_dir), "--baseline", str(self.baseline_path), *extra]
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cpr.main(argv)
        return code, buf.getvalue().strip()


class StaticFallbackTests(_GateTestBase):
    """With no rolling window, the static baseline_fps + pct bands govern."""

    def test_pass_within_band(self) -> None:
        self._write_result(_omni_doc(int(BASELINE_FPS * 0.97)))
        self._write_baseline(_baseline())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_PASS)
        self.assertIn("RESULT=PASS", out)
        self.assertIn("thresholds=static_baseline", out)

    def test_improvement_passes(self) -> None:
        self._write_result(_omni_doc(int(BASELINE_FPS * 1.08)))
        self._write_baseline(_baseline())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_PASS)
        self.assertIn("delta_pct=+8.00", out)

    def test_warn_band_is_advisory(self) -> None:
        # 7% drop: past the 5% warn band, short of the 10% block band -> WARN, exit 0.
        self._write_result(_omni_doc(int(BASELINE_FPS * 0.93)))
        self._write_baseline(_baseline())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_PASS)
        self.assertIn("RESULT=WARN", out)

    def test_block_on_large_regression(self) -> None:
        self._write_result(_omni_doc(int(BASELINE_FPS * 0.80)))
        self._write_baseline(_baseline())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_BLOCK)
        self.assertIn("RESULT=BLOCK", out)
        self.assertIn("kind=regression", out)


class WindowThresholdTests(_GateTestBase):
    """With >= MIN_WINDOW samples, median+MAD over the window governs."""

    def test_uses_window_when_available(self) -> None:
        self._write_result(_omni_doc(10000))
        self._write_baseline(_baseline(fps=999999))  # static would BLOCK; window must win
        self._write_window([10000, 10010, 9990, 10005, 9995])
        code, out = self._run("--history-dir", str(self.history_dir))
        self.assertEqual(code, cpr.EXIT_PASS)
        self.assertIn("thresholds=window(n=5)", out)
        self.assertIn("center_fps=10000", out)

    def test_block_below_window(self) -> None:
        self._write_result(_omni_doc(8000))
        self._write_baseline(_baseline(fps=10000))
        self._write_window([10000, 10010, 9990, 10005, 9995])
        code, out = self._run("--history-dir", str(self.history_dir))
        self.assertEqual(code, cpr.EXIT_BLOCK)
        self.assertIn("kind=regression", out)

    def test_small_window_falls_back_to_static(self) -> None:
        self._write_result(_omni_doc(9700))
        self._write_baseline(_baseline(fps=10000))
        self._write_window([10000, 9990])  # < MIN_WINDOW
        code, out = self._run("--history-dir", str(self.history_dir))
        self.assertIn("thresholds=static_baseline", out)


class SteadyMetricTests(_GateTestBase):
    """D6: the gating KPI is mean effective FPS after dropping warmup_frames."""

    def test_drops_default_warmup(self) -> None:
        # Two tiny warm-up frames then steady at the baseline; dropping 2 -> PASS.
        eff = [1.0, 1.0] + [BASELINE_FPS] * 100
        self._write_result(_eff_doc(eff))
        self._write_baseline(_baseline())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_PASS)
        self.assertIn(f"measured_fps={BASELINE_FPS}", out)

    def test_task_warmup_override_honored(self) -> None:
        # First 5 frames are slow; with warmup_frames=5 they are excluded -> PASS.
        eff = [1.0] * 5 + [BASELINE_FPS] * 100
        self._write_result(_eff_doc(eff))
        self._write_baseline(_baseline(warmup=5))
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_PASS)
        self.assertIn("warmup_frames=5", out)

    def test_scalar_fallback_when_no_array(self) -> None:
        self._write_result(_eff_doc(None, scalar=BASELINE_FPS))
        self._write_baseline(_baseline())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_PASS)

    def test_num_frames_truncation(self) -> None:
        # Trailing frames beyond num_frames are ignored so a longer run stays comparable.
        eff = [1.0, 1.0] + [BASELINE_FPS] * 8 + [1.0] * 100
        self._write_result(_eff_doc(eff))
        self._write_baseline(_baseline(num_frames=10))
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_PASS)
        self.assertIn(f"measured_fps={BASELINE_FPS}", out)


class DebugAndWallTests(_GateTestBase):
    """Advisory KPIs: outlier index/magnitude and the wall-clock signal."""

    def test_outlier_index_and_magnitude(self) -> None:
        steps = [300.0, 140.0] + [10.0] * 20
        steps[7] = 30.0  # 3x the steady median at steady-frame index 5 (after dropping 2)
        self._write_result(_eff_doc([BASELINE_FPS] * len(steps), step_times=steps))
        self._write_baseline(_baseline())
        _, out = self._run()
        self.assertIn("outlier_count=1", out)
        self.assertIn("outlier_idx=5", out)
        self.assertIn("outlier_mag_x=3", out)

    def test_wall_signal_reported(self) -> None:
        self._write_result(_omni_doc(BASELINE_FPS))
        self._write_baseline(_baseline())
        self._write_window([BASELINE_FPS] * 5, wall=[100, 101, 99, 100, 100])
        _, out = self._run("--history-dir", str(self.history_dir), "--measured-wall-s", "100")
        self.assertIn("wall_center_s=100", out)
        self.assertIn("wall_delta_pct=", out)

    def test_wall_flag_when_slow(self) -> None:
        self._write_result(_omni_doc(BASELINE_FPS))
        self._write_baseline(_baseline())
        self._write_window([BASELINE_FPS] * 5, wall=[100, 101, 99, 100, 100])
        _, out = self._run("--history-dir", str(self.history_dir), "--measured-wall-s", "130")
        self.assertIn("wall_flag=slow", out)


class WarmupGuardTests(_GateTestBase):
    """Advisory warm-up guardrail: flag a hot first kept frame (stale warmup_frames)."""

    def test_flag_when_first_kept_frame_hot(self) -> None:
        # Drop 2 warm-up frames; the first KEPT step is 5x the steady median -> flag.
        steps = [300.0, 280.0] + [50.0] + [10.0] * 30
        self._write_result(_eff_doc([BASELINE_FPS] * len(steps), step_times=steps))
        self._write_baseline(_baseline())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_PASS)  # advisory only -- never changes the verdict
        self.assertIn("warmup_flag=", out)

    def test_no_flag_when_first_kept_frame_steady(self) -> None:
        steps = [300.0, 280.0] + [10.0] * 30
        self._write_result(_eff_doc([BASELINE_FPS] * len(steps), step_times=steps))
        self._write_baseline(_baseline())
        _, out = self._run()
        self.assertNotIn("warmup_flag=", out)


class TailKpiTests(_GateTestBase):
    """Opt-in advisory tail signal: WARN (never BLOCK) on a high p99/median ratio."""

    def _doc_with_p99(self, ratio: float) -> list:
        # Steady median 10ms; a top ~3% of steps at ratio*median pushes p99/median ~= ratio.
        steps = [10.0] * 100
        for i in (50, 51, 52):  # >= 2% so the p99 index lands on a spike
            steps[i] = 10.0 * ratio
        return _eff_doc([BASELINE_FPS] * len(steps), step_times=steps)

    def test_tail_warn_when_over_ceiling(self) -> None:
        self._write_result(self._doc_with_p99(3.0))
        self._write_baseline(_baseline())
        self._write_overrides({TASK: {GPU: {"tail_p99_warn": 1.5}}})
        code, out = self._run("--overrides", str(self.overrides_path))
        self.assertEqual(code, cpr.EXIT_PASS)  # WARN is advisory -> exit 0
        self.assertIn("RESULT=WARN", out)
        self.assertIn("reason=tail", out)
        self.assertIn("tail_flag=", out)

    def test_no_tail_warn_under_ceiling(self) -> None:
        self._write_result(self._doc_with_p99(1.2))
        self._write_baseline(_baseline())
        self._write_overrides({TASK: {GPU: {"tail_p99_warn": 2.0}}})
        code, out = self._run("--overrides", str(self.overrides_path))
        self.assertEqual(code, cpr.EXIT_PASS)
        self.assertIn("RESULT=PASS", out)
        self.assertNotIn("tail_flag=", out)

    def test_tail_disabled_by_default(self) -> None:
        # Without the override, a spiky run still PASSes (no tail gating).
        self._write_result(self._doc_with_p99(5.0))
        self._write_baseline(_baseline())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_PASS)
        self.assertIn("RESULT=PASS", out)

    def test_fps_block_wins_over_tail(self) -> None:
        # A real FPS regression must BLOCK even with a tail ceiling set.
        steps = [10.0] * 100
        steps[50] = 50.0
        self._write_result(_eff_doc([int(BASELINE_FPS * 0.8)] * len(steps), step_times=steps))
        self._write_baseline(_baseline())
        self._write_overrides({TASK: {GPU: {"tail_p99_warn": 1.5}}})
        code, out = self._run("--overrides", str(self.overrides_path))
        self.assertEqual(code, cpr.EXIT_BLOCK)
        self.assertIn("kind=regression", out)


class OverrideTests(_GateTestBase):
    """Manual overrides (committed with the PR) adjust or bypass the gate."""

    def test_skip_forces_pass(self) -> None:
        self._write_result(_omni_doc(1))  # would BLOCK
        self._write_baseline(_baseline())
        self._write_overrides({TASK: {GPU: {"skip": True}}})
        code, out = self._run("--overrides", str(self.overrides_path))
        self.assertEqual(code, cpr.EXIT_PASS)
        self.assertIn("skipped_by_override", out)

    def test_pin_center(self) -> None:
        self._write_result(_omni_doc(120000))
        self._write_baseline(_baseline())
        self._write_overrides({TASK: {GPU: {"pin_center_fps": 120000}}})
        code, out = self._run("--overrides", str(self.overrides_path))
        self.assertEqual(code, cpr.EXIT_PASS)
        self.assertIn("thresholds=override_pin", out)
        self.assertIn("center_fps=120000", out)

    def test_k_block_override_tightens(self) -> None:
        # A 4% drop passes the default k_block but a tightened spread should BLOCK.
        self._write_result(_omni_doc(int(BASELINE_FPS * 0.96)))
        self._write_baseline(_baseline())
        self._write_window([BASELINE_FPS] * 5)
        self._write_overrides({TASK: {GPU: {"k_block": 1.0, "min_spread_pct": 1.0}}})
        code, _ = self._run("--history-dir", str(self.history_dir), "--overrides", str(self.overrides_path))
        self.assertEqual(code, cpr.EXIT_BLOCK)


class HardFailureTests(_GateTestBase):
    """Structural problems map to BLOCK/hard_failure (exit 2)."""

    def test_no_results_file(self) -> None:
        self._write_baseline(_baseline())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)
        self.assertIn("kind=hard_failure", out)
        self.assertIn("no_results_found", out)

    def test_malformed_json(self) -> None:
        self._write_result("{not valid json")
        self._write_baseline(_baseline())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)
        self.assertIn("malformed_json", out)

    def test_missing_task(self) -> None:
        self._write_result(_omni_doc(BASELINE_FPS))
        self._write_baseline({"other-task": {"per_gpu": {GPU: {"baseline_fps": 1}}}})
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)
        self.assertIn("missing_baseline_task", out)

    def test_missing_baseline_fps(self) -> None:
        self._write_result(_omni_doc(BASELINE_FPS))
        self._write_baseline({TASK: {"per_gpu": {GPU: {"warn_pct": 5.0}}}})
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)
        self.assertIn("missing_baseline_field", out)

    def test_gpu_mismatch(self) -> None:
        self._write_result(_omni_doc(BASELINE_FPS, gpu="NVIDIA Other"))
        self._write_baseline(_baseline())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)
        self.assertIn("baseline_gpu_mismatch", out)

    def test_unknown_gpu_without_override(self) -> None:
        self._write_result(_omni_doc(BASELINE_FPS, gpu=None))
        self._write_baseline(_baseline())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)
        self.assertIn("unknown_gpu", out)

    def test_zero_metric(self) -> None:
        self._write_result(_omni_doc(0))
        self._write_baseline(_baseline())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)
        self.assertIn("missing_metric", out)

    def test_string_metric(self) -> None:
        self._write_result(_omni_doc("nope"))
        self._write_baseline(_baseline())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)

    def test_nan_metric(self) -> None:
        path = self.results_dir / f"benchmark_non_rl_{TASK}.json"
        path.write_text(
            '{"runtime": {"' + cpr.METRIC_NAME + '": NaN}, '
            '"hardware_info": {"gpu_current_device": 0, "gpu_devices": {"0": {"name": "' + GPU + '"}}}}',
            encoding="utf-8",
        )
        self._write_baseline(_baseline())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)


def _info_phase(
    task: str = TASK, num_envs: int = 512, seed: int = 42, num_frames: int = 300, presets: str | None = None
) -> dict:
    """A ``benchmark_info`` phase echoing the run config (for config-assert tests)."""
    meta = [
        {"name": "benchmark_non_rl benchmark_info task", "data": task},
        {"name": "benchmark_non_rl benchmark_info seed", "data": seed},
        {"name": "benchmark_non_rl benchmark_info num_envs", "data": num_envs},
        {"name": "benchmark_non_rl benchmark_info num_frames", "data": num_frames},
    ]
    if presets is not None:
        meta.append({"name": "benchmark_non_rl benchmark_info presets", "data": presets})
    return {"phase_name": "benchmark_info", "measurements": [], "metadata": meta}


class ConfigAssertTests(_GateTestBase):
    """The run's self-reported config must match baseline.json, else hard_failure."""

    def _doc_with_info(self, info: dict) -> list:
        doc = _eff_doc([BASELINE_FPS] * 50)
        doc.append(info)
        return doc

    def test_matching_config_passes(self) -> None:
        self._write_result(self._doc_with_info(_info_phase(num_envs=512, seed=42, num_frames=300)))
        self._write_baseline(
            {TASK: {"num_envs": 512, "seed": 42, "num_frames": 300, "per_gpu": {GPU: {"baseline_fps": BASELINE_FPS}}}}
        )
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_PASS)

    def test_num_envs_mismatch_blocks(self) -> None:
        self._write_result(self._doc_with_info(_info_phase(num_envs=4096)))
        self._write_baseline({TASK: {"num_envs": 512, "per_gpu": {GPU: {"baseline_fps": BASELINE_FPS}}}})
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)
        self.assertIn("config_mismatch", out)
        self.assertIn("num_envs(ran=4096,want=512)", out)

    def test_too_few_frames_blocks(self) -> None:
        self._write_result(self._doc_with_info(_info_phase(num_frames=100)))
        self._write_baseline({TASK: {"num_frames": 300, "per_gpu": {GPU: {"baseline_fps": BASELINE_FPS}}}})
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)
        self.assertIn("config_mismatch", out)

    def test_no_benchmark_info_is_noop(self) -> None:
        # OmniPerf / legacy results carry no benchmark_info -> assertion is skipped.
        self._write_result(_omni_doc(BASELINE_FPS))
        self._write_baseline({TASK: {"num_envs": 512, "per_gpu": {GPU: {"baseline_fps": BASELINE_FPS}}}})
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_PASS)

    def test_matching_presets_pass(self) -> None:
        self._write_result(self._doc_with_info(_info_phase(presets="newton_mjwarp")))
        self._write_baseline(
            {TASK: {"benchmark_args": ["physics=newton_mjwarp"], "per_gpu": {GPU: {"baseline_fps": BASELINE_FPS}}}}
        )
        code, _ = self._run()
        self.assertEqual(code, cpr.EXIT_PASS)

    def test_wrong_backend_blocks(self) -> None:
        # Baseline expects Newton but the run reported PhysX -> a different KPI, hard_failure.
        self._write_result(self._doc_with_info(_info_phase(presets="physx")))
        self._write_baseline(
            {TASK: {"benchmark_args": ["physics=newton_mjwarp"], "per_gpu": {GPU: {"baseline_fps": BASELINE_FPS}}}}
        )
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)
        self.assertIn("config_mismatch", out)
        self.assertIn("presets(", out)
        self.assertIn("missing=newton_mjwarp", out)

    def test_multi_preset_subset_match(self) -> None:
        # Each expected token (physx + renderer) must appear in the comma-joined presets.
        self._write_result(self._doc_with_info(_info_phase(presets="physx,isaacsim_rtx_renderer")))
        self._write_baseline(
            {
                TASK: {
                    "benchmark_args": ["physics=physx", "presets=isaacsim_rtx_renderer"],
                    "per_gpu": {GPU: {"baseline_fps": BASELINE_FPS}},
                }
            }
        )
        code, _ = self._run()
        self.assertEqual(code, cpr.EXIT_PASS)

    def test_presets_unreported_is_noop(self) -> None:
        # Older results omit presets -> we don't assert (no false BLOCK).
        self._write_result(self._doc_with_info(_info_phase(presets=None)))
        self._write_baseline(
            {TASK: {"benchmark_args": ["physics=newton_mjwarp"], "per_gpu": {GPU: {"baseline_fps": BASELINE_FPS}}}}
        )
        code, _ = self._run()
        self.assertEqual(code, cpr.EXIT_PASS)


class VariantKeyTests(_GateTestBase):
    """A '<gym id>@<backend>' gate key resolves via task_id for glob + config check."""

    VARIANT = f"{TASK}@newton"

    def _run_variant(self, *extra: str) -> tuple[int, str]:
        argv = [
            "--task",
            self.VARIANT,
            "--task-id",
            TASK,
            "--results-dir",
            str(self.results_dir),
            "--baseline",
            str(self.baseline_path),
            *extra,
        ]
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cpr.main(argv)
        return code, buf.getvalue().strip()

    def test_variant_resolves_and_passes(self) -> None:
        # Result file is named by the gym id; baseline is keyed by the variant key.
        self._write_result(_eff_doc([BASELINE_FPS] * 50), name=f"benchmark_non_rl_{TASK}_x.json")
        self._write_baseline({self.VARIANT: {"task_id": TASK, "per_gpu": {GPU: {"baseline_fps": BASELINE_FPS}}}})
        code, out = self._run_variant()
        self.assertEqual(code, cpr.EXIT_PASS)
        self.assertIn(f"task={self.VARIANT}", out)

    def test_task_id_defaults_to_prefix_before_at(self) -> None:
        # Without --task-id, the gym id is inferred as the part before "@".
        self._write_result(_eff_doc([BASELINE_FPS] * 50), name=f"benchmark_non_rl_{TASK}_x.json")
        self._write_baseline({self.VARIANT: {"task_id": TASK, "per_gpu": {GPU: {"baseline_fps": BASELINE_FPS}}}})
        argv = [
            "--task",
            self.VARIANT,
            "--results-dir",
            str(self.results_dir),
            "--baseline",
            str(self.baseline_path),
        ]
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cpr.main(argv)
        self.assertEqual(code, cpr.EXIT_PASS)


class HelperTests(unittest.TestCase):
    """Direct unit tests for helpers where edge cases are easier to express."""

    def test_steady_fps_drops_warmup(self) -> None:
        result = {"runtime": {cpr.FRAMETIMES_NAME: {cpr.EFF_FPS_ARRAY: [1.0, 1.0, 100.0, 100.0]}}}
        self.assertEqual(cpr.steady_fps(result, warmup_frames=2), 100.0)

    def test_steady_fps_scalar_fallback(self) -> None:
        self.assertEqual(cpr.steady_fps({"runtime": {cpr.METRIC_NAME: 1234}}, warmup_frames=2), 1234.0)

    def test_steady_fps_rejects_empty(self) -> None:
        with self.assertRaises(cpr.CompareError):
            cpr.steady_fps({"runtime": {}}, warmup_frames=2)

    def test_median_mad(self) -> None:
        center, mad = cpr._median_mad([10.0, 12.0, 14.0])
        self.assertEqual(center, 12.0)
        self.assertEqual(mad, 2.0)

    def test_match_gpu_substring(self) -> None:
        key, _ = cpr._match_gpu({"L40": {"baseline_fps": 1.0}}, "NVIDIA L40")
        self.assertEqual(key, "L40")

    def test_overrides_precedence(self) -> None:
        ov = cpr._overrides_for({"_defaults": {"k_warn": 3}, TASK: {"k_warn": 4, GPU: {"k_warn": 5}}}, TASK, GPU)
        self.assertEqual(ov["k_warn"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
