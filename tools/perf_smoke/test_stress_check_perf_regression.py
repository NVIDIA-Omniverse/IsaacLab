# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adversarial / stress tests for ``check_perf_regression.py`` (the perf-smoke comparator).

The companion suite ``test_check_perf_regression.py`` exercises the *happy paths*
of the comparator. This file deliberately attacks the edges -- malformed history,
hostile GPU strings, degenerate baselines, truncated runs, poisoned inputs -- to
surface correctness and safety gaps. Findings are written up in
``POC_LIMITATIONS_REPORT.md``.

Two kinds of tests live here:

* **Lock tests** assert the comparator's *current, observed* behavior on an edge
  case (so a future change is forced to acknowledge it).
* **``@unittest.expectedFailure`` tests** assert the behavior we *want* but the
  comparator does **not** yet deliver. They pass as "expected failures" today; if
  the bug is ever fixed, they flip to "unexpected success" and prompt removing the
  marker. Each carries a ``BUG:`` note keyed to the report.

Run directly::

    python3 tools/perf_smoke/test_stress_check_perf_regression.py
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

TASK = "Isaac-Cartpole-v0"
GPU = "NVIDIA L40S"
BASELINE_FPS = 100000


def _omni(fps: float | int | str | None, gpu: str | None = GPU) -> dict:
    doc: dict = {"runtime": {cpr.METRIC_NAME: fps}}
    if gpu is not None:
        doc["hardware_info"] = {"gpu_current_device": 0, "gpu_devices": {"0": {"name": gpu}}}
    return doc


def _eff(eff: list[float] | None, gpu: str = GPU, steps: list[float] | None = None) -> list:
    ft: dict = {}
    if eff is not None:
        ft[cpr.EFF_FPS_ARRAY] = eff
    if steps is not None:
        ft[cpr.STEP_MS_ARRAY] = steps
    return [
        {
            "phase_name": "runtime",
            "measurements": [{"name": f"benchmark_non_rl runtime {cpr.FRAMETIMES_NAME}", "value": ft}],
            "metadata": [],
        },
        {
            "phase_name": "hardware_info",
            "measurements": [],
            "metadata": [
                {"name": "benchmark_non_rl hardware_info gpu_current_device", "data": 0},
                {"name": "benchmark_non_rl hardware_info gpu_devices", "data": {"0": {"name": gpu}}},
            ],
        },
    ]


def _info(**fields: object) -> dict:
    meta = [{"name": f"benchmark_non_rl benchmark_info {k}", "data": v} for k, v in fields.items()]
    return {"phase_name": "benchmark_info", "measurements": [], "metadata": meta}


def _baseline(
    fps: float = BASELINE_FPS, warn: float = 5.0, block: float = 10.0, gpu_key: str = GPU, **extra: object
) -> dict:
    entry: dict = {"per_gpu": {gpu_key: {"baseline_fps": fps, "warn_pct": warn, "max_regression_pct": block}}}
    entry.update(extra)
    return {TASK: entry}


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.results_dir = self.tmp / "out"
        self.results_dir.mkdir()
        self.baseline_path = self.tmp / "baseline.json"
        self.history_dir = self.tmp / "history"
        self.overrides_path = self.tmp / "overrides.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_result(self, doc, name: str | None = None) -> None:
        path = self.results_dir / (name or f"benchmark_non_rl_{TASK}_x.json")
        path.write_text(doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8")

    def _write_baseline(self, doc: dict) -> None:
        self.baseline_path.write_text(json.dumps(doc), encoding="utf-8")

    def _write_window_raw(self, samples: list[dict], gpu_key: str = GPU) -> None:
        self.history_dir.mkdir(exist_ok=True)
        safe = f"{TASK}__{gpu_key}".replace(" ", "_")
        # allow_nan=True keeps this realistic: json.dump writes NaN/Infinity by default.
        (self.history_dir / f"{safe}.json").write_text(json.dumps({"samples": samples}), encoding="utf-8")

    def _write_overrides(self, doc: dict) -> None:
        self.overrides_path.write_text(json.dumps(doc), encoding="utf-8")

    def _run(self, *extra: str, task: str = TASK):
        argv = ["--task", task, "--results-dir", str(self.results_dir), "--baseline", str(self.baseline_path), *extra]
        buf = io.StringIO()
        exc = None
        with redirect_stdout(buf):
            try:
                code = cpr.main(argv)
            except Exception as e:  # noqa: BLE001 -- we WANT to see uncaught exceptions
                code, exc = None, e
        return code, buf.getvalue().strip(), exc


# --------------------------------------------------------------------------- A
class DegenerateBaselineTests(_Base):
    """A zero / tiny baseline center must not crash the comparator."""

    @unittest.expectedFailure
    def test_zero_center_should_be_hard_failure_not_crash(self) -> None:
        # BUG A: baseline_fps=0 -> delta_pct = (m-0)/0 -> uncaught ZeroDivisionError.
        # Desired: a structural BLOCK/hard_failure, never a Python traceback.
        self._write_result(_omni(50000))
        self._write_baseline(_baseline(fps=0))
        code, _out, exc = self._run()
        self.assertIsNone(exc, f"comparator crashed instead of degrading: {exc!r}")
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)

    def test_zero_center_currently_crashes(self) -> None:
        # Lock the present (bad) behavior so the report stays accurate.
        self._write_result(_omni(50000))
        self._write_baseline(_baseline(fps=0))
        _code, _out, exc = self._run()
        self.assertIsInstance(exc, ZeroDivisionError)


# --------------------------------------------------------------------------- B
class GpuMatchingTests(_Base):
    """Substring GPU matching conflates distinct GPUs."""

    @unittest.expectedFailure
    def test_l40_must_not_match_l40s(self) -> None:
        # BUG B: result GPU 'NVIDIA L40' substring-matches baseline 'NVIDIA L40S'
        # (gpu_key in key), so an L40 run is judged against an L40S window.
        self._write_result(_omni(50000, gpu="NVIDIA L40"))
        self._write_baseline(_baseline(fps=50000, gpu_key="NVIDIA L40S"))
        code, out, _ = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE, f"expected GPU mismatch, got: {out}")

    @unittest.expectedFailure
    def test_desktop_4090_must_not_match_laptop_4090(self) -> None:
        # BUG B: 'RTX 4090' is a substring of 'RTX 4090 Laptop GPU' -> false match
        # between two very differently-performing GPUs.
        self._write_result(_omni(50000, gpu="NVIDIA GeForce RTX 4090"))
        self._write_baseline(_baseline(fps=50000, gpu_key="NVIDIA GeForce RTX 4090 Laptop GPU"))
        code, _out, _ = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)

    def test_substring_match_is_current_behavior(self) -> None:
        self._write_result(_omni(50000, gpu="NVIDIA L40"))
        self._write_baseline(_baseline(fps=50000, gpu_key="NVIDIA L40S"))
        code, out, _ = self._run()
        self.assertEqual(code, cpr.EXIT_PASS)
        self.assertIn("gpu=NVIDIA L40S", out)


# --------------------------------------------------------------------------- C
class TruncatedRunTests(_Base):
    """A run with fewer frames than warmup_frames silently skips warm-up exclusion."""

    @unittest.expectedFailure
    def test_short_run_should_not_silently_keep_warmup(self) -> None:
        # BUG C: only 3 frames exist but warmup_frames=60. steady_fps keeps the whole
        # (warm-up-polluted) window instead of erroring, so a truncated/crashed run is
        # accepted as a valid steady measurement.
        self._write_result(_eff([10.0, 10.0, 10.0]))  # all warm-up frames
        self._write_baseline(_baseline(fps=10, warmup_frames=60, num_frames=300))
        code, out, _ = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE, f"truncated run accepted as valid: {out}")

    def test_short_run_currently_passes(self) -> None:
        self._write_result(_eff([10.0, 10.0, 10.0]))
        self._write_baseline(_baseline(fps=10, warmup_frames=60, num_frames=300))
        code, _out, _ = self._run()
        self.assertEqual(code, cpr.EXIT_PASS)


# --------------------------------------------------------------------------- D
class StaticBandFidelityTests(_Base):
    """The static fallback cannot honor warn_pct and max_regression_pct independently."""

    @unittest.expectedFailure
    def test_block_band_honored_when_not_2x_warn(self) -> None:
        # BUG D: warn_pct=5, max_regression_pct=8. A -9% drop is past the configured
        # 8% block band and should BLOCK -- but the spread is driven by the warn band,
        # so the effective block floor slips to -10% and the run only WARNs.
        self._write_result(_omni(int(BASELINE_FPS * 0.91)))  # -9%
        self._write_baseline(_baseline(warn=5.0, block=8.0))
        code, out, _ = self._run()
        self.assertEqual(code, cpr.EXIT_BLOCK, f"configured 8% block band not honored: {out}")

    @unittest.expectedFailure
    def test_warn_band_honored_when_block_is_large(self) -> None:
        # BUG D (other direction): warn_pct=5, max_regression_pct=20. A -8% drop is
        # past the configured 5% warn band and should WARN -- but the block-derived
        # spread widens the warn floor to -10%, so this PASSes silently.
        self._write_result(_omni(int(BASELINE_FPS * 0.92)))  # -8%
        self._write_baseline(_baseline(warn=5.0, block=20.0))
        code, out, _ = self._run()
        self.assertIn("RESULT=WARN", out, f"configured 5% warn band not honored: {out}")


# --------------------------------------------------------------------------- E
class PoisonedHistoryTests(_Base):
    """A single corrupt history sample silently disables the gate."""

    @unittest.expectedFailure
    def test_nan_in_window_should_not_blind_the_gate(self) -> None:
        # BUG E: one NaN fps at the median position -> median=nan -> all thresholds nan
        # -> every '<' comparison is False -> the task can never BLOCK again. Silent.
        # (NaN at index 2 of a 5-sample window reliably poisons the median; whether a
        # NaN poisons at all is position-dependent, which is itself the bug.)
        self._write_result(_omni(1))  # a catastrophic regression
        self._write_baseline(_baseline(fps=BASELINE_FPS))
        self._write_window_raw([{"fps": BASELINE_FPS}] * 2 + [{"fps": float("nan")}] + [{"fps": BASELINE_FPS}] * 2)
        code, out, _ = self._run("--history-dir", str(self.history_dir))
        self.assertNotIn("center_fps=nan", out)
        self.assertEqual(code, cpr.EXIT_BLOCK, f"NaN window let a real regression PASS: {out}")

    def test_nan_window_currently_passes_everything(self) -> None:
        # Lock the current behavior: a median-position NaN disables the gate (PASS).
        self._write_result(_omni(1))
        self._write_baseline(_baseline(fps=BASELINE_FPS))
        self._write_window_raw([{"fps": BASELINE_FPS}] * 2 + [{"fps": float("nan")}] + [{"fps": BASELINE_FPS}] * 2)
        code, out, _ = self._run("--history-dir", str(self.history_dir))
        self.assertEqual(code, cpr.EXIT_PASS)
        self.assertIn("center_fps=nan", out)

    def test_nan_effect_is_position_dependent(self) -> None:
        # The same poisoned window judged with the NaN at a tail position yields a
        # REAL median -> the gate works. Identical data, different order, opposite
        # verdict: NaN handling is undefined, not merely lenient.
        self._write_result(_omni(1))
        self._write_baseline(_baseline(fps=BASELINE_FPS))
        self._write_window_raw([{"fps": BASELINE_FPS}] * 4 + [{"fps": float("nan")}])  # NaN at tail
        code, out, _ = self._run("--history-dir", str(self.history_dir))
        self.assertEqual(code, cpr.EXIT_BLOCK)
        self.assertNotIn("center_fps=nan", out)


# --------------------------------------------------------------------------- F
class PoisonedMeasurementTests(_Base):
    """Negative per-frame FPS values are averaged into the KPI instead of rejected."""

    @unittest.expectedFailure
    def test_negative_frames_should_be_caught(self) -> None:
        # BUG F: steady_fps only checks the *final mean* > 0; individual negative
        # (impossible) per-frame FPS values silently drag the mean instead of failing.
        arr = [BASELINE_FPS] * 100 + [-BASELINE_FPS] * 5
        self._write_result(_eff(arr))
        self._write_baseline(_baseline(fps=BASELINE_FPS))
        code, out, _ = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE, f"negative frames silently averaged: {out}")


# --------------------------------------------------------------------------- G
class GlobCollisionTests(_Base):
    """The result glob is prefix-greedy; sibling-prefixed task names collide."""

    def test_sibling_prefix_causes_false_hard_failure(self) -> None:
        # BUG G: glob 'benchmark_non_rl_Isaac-Cartpole-v0*.json' also matches a sibling
        # task 'Isaac-Cartpole-v0-Camera'. Without --allow-multiple this is a spurious
        # multiple_results hard_failure (a FALSE BLOCK) even though the real result is present.
        self._write_result(_omni(BASELINE_FPS), name="benchmark_non_rl_Isaac-Cartpole-v0_a.json")
        self._write_result(_omni(1000), name="benchmark_non_rl_Isaac-Cartpole-v0-Camera_b.json")
        self._write_baseline(_baseline())
        code, out, _ = self._run()
        # Lock the (undesirable) current behavior.
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)
        self.assertIn("multiple_results", out)


# --------------------------------------------------------------------------- H
class QuarantineTests(_Base):
    """`skip` cannot quarantine a task that crashed (no result produced)."""

    @unittest.expectedFailure
    def test_skip_should_quarantine_a_broken_task(self) -> None:
        # BUG H: the docs sell `skip:true` as the escape hatch for a "temporarily
        # flaky" task -- but flaky tasks usually crash, and hard_failure (missing
        # result) is evaluated BEFORE the skip override, so skip cannot rescue them.
        self._write_baseline(_baseline())  # NOTE: no result file written -> crash
        self._write_overrides({TASK: {GPU: {"skip": True}}})
        code, out, _ = self._run("--overrides", str(self.overrides_path), "--gpu-override", GPU)
        self.assertEqual(code, cpr.EXIT_PASS, f"skip did not quarantine a broken task: {out}")


# --------------------------------------------------------------------------- I
class PinCenterTests(_Base):
    """Pinning a center without pinning spread yields nonsensical bands."""

    @unittest.expectedFailure
    def test_pin_center_should_scale_spread(self) -> None:
        # BUG I: pin_center_fps=300000 but spread is still computed from the OLD window
        # center (~100000), so the band becomes ~0.5% of the new center. A run within
        # 5% of the intended new center then falsely BLOCKs -- the opposite of the
        # override's purpose (accepting an intended perf change).
        self._write_result(_omni(285000))  # -5% vs the pinned center
        self._write_baseline(_baseline(fps=BASELINE_FPS))
        self._write_window_raw([{"fps": BASELINE_FPS}] * 5)
        self._write_overrides({TASK: {GPU: {"pin_center_fps": 300000}}})
        code, out, _ = self._run("--history-dir", str(self.history_dir), "--overrides", str(self.overrides_path))
        self.assertEqual(code, cpr.EXIT_PASS, f"pinned-center bands collapsed: {out}")


# --------------------------------------------------------------------------- J
class ConfigAssertTypeTests(_Base):
    """Config assertion is skipped when the backend serializes numbers as strings."""

    @unittest.expectedFailure
    def test_string_num_envs_should_still_be_checked(self) -> None:
        # BUG J: benchmark_info.num_envs reported as "4096" (str). _assert_run_config
        # guards on isinstance(got,(int,float)), so a string slips past unchecked and a
        # config change is silently misread as a perf number.
        doc = _eff([BASELINE_FPS] * 50)
        doc.append(_info(num_envs="4096"))
        self.baseline_path.write_text(
            json.dumps({TASK: {"num_envs": 512, "per_gpu": {GPU: {"baseline_fps": BASELINE_FPS}}}}), encoding="utf-8"
        )
        self._write_result(doc)
        code, out, _ = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE, f"string num_envs bypassed config check: {out}")


# --------------------------------------------------------------------------- misc
class BoundaryTests(_Base):
    """Smaller, lower-severity edge observations worth locking down."""

    def test_block_floor_is_exclusive(self) -> None:
        # A measurement landing *exactly* on the block floor does not BLOCK (strict <).
        # Documented here so the off-by-epsilon boundary is intentional, not accidental.
        self._write_result(_omni(90000))  # exactly -10% with a 10% block band
        self._write_baseline(_baseline(warn=5.0, block=10.0))
        code, _out, _ = self._run()
        self.assertEqual(code, cpr.EXIT_PASS)

    def test_window_warmup_metadata_is_ignored(self) -> None:
        # The window file records the warmup_frames it was computed with, but the
        # comparator never checks it against the baseline's current warmup_frames, so a
        # later warmup change compares new-warmup measurements to old-warmup history.
        self._write_result(_omni(BASELINE_FPS))
        self._write_baseline(_baseline(fps=BASELINE_FPS, warmup_frames=2))
        self.history_dir.mkdir(exist_ok=True)
        safe = f"{TASK}__{GPU}".replace(" ", "_")
        (self.history_dir / f"{safe}.json").write_text(
            json.dumps({"warmup_frames": 60, "samples": [{"fps": BASELINE_FPS}] * 5}), encoding="utf-8"
        )
        code, out, _ = self._run("--history-dir", str(self.history_dir))
        self.assertEqual(code, cpr.EXIT_PASS)
        self.assertIn("warmup_frames=2", out)  # uses baseline's, ignores window's 60


if __name__ == "__main__":
    unittest.main(verbosity=2)
