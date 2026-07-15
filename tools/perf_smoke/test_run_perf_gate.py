# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ``run_perf_gate.py`` (the Phase 1 perf-smoke orchestrator).

Covers the pure-logic surface that needs no GPU: launch-config resolution from
``baseline.json``, benchmark command construction, the warm-cache env overlay,
comparator ``RESULT=`` label parsing, and the worst-verdict aggregation in
``main`` via ``--dry-run``.

Stdlib ``unittest``; also collectable by pytest via this directory's ``pytest.ini``.
Run directly::

    python3 tools/perf_smoke/test_run_perf_gate.py
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
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_perf_gate as gate  # noqa: E402

_BASELINE = {
    "_meta": {"note": "ignored"},
    "Isaac-Cartpole-v0": {
        "num_envs": 4096,
        "num_frames": 300,
        "seed": 42,
        "per_gpu": {"NVIDIA L40S": {"baseline_fps": 100000.0}},
    },
    "Isaac-Cartpole-v0@newton": {
        "task_id": "Isaac-Cartpole-v0",
        "num_envs": 4096,
        "benchmark_args": ["presets=newton"],
        "per_gpu": {"NVIDIA L40S": {"baseline_fps": 90000.0}},
    },
}


class TaskRunConfig(unittest.TestCase):
    def test_defaults_filled_for_absent_fields(self):
        cfg = gate._task_run_config(_BASELINE, "Isaac-Cartpole-v0")
        self.assertEqual(cfg["task_id"], "Isaac-Cartpole-v0")
        self.assertEqual(cfg["num_envs"], 4096)
        self.assertEqual(cfg["num_frames"], 300)
        self.assertEqual(cfg["seed"], 42)
        self.assertEqual(cfg["benchmark_args"], [])

    def test_variant_key_resolves_to_gym_task_id(self):
        cfg = gate._task_run_config(_BASELINE, "Isaac-Cartpole-v0@newton")
        self.assertEqual(cfg["task_id"], "Isaac-Cartpole-v0")
        self.assertEqual(cfg["benchmark_args"], ["presets=newton"])
        # seed/num_frames fall back to calibration defaults when unset.
        self.assertEqual(cfg["num_frames"], 300)
        self.assertEqual(cfg["seed"], 42)

    def test_missing_task_raises_keyerror(self):
        with self.assertRaises(KeyError):
            gate._task_run_config(_BASELINE, "Nope")


class BenchmarkCmd(unittest.TestCase):
    def test_cmd_carries_task_id_frames_seed_and_json_backend(self):
        cfg = gate._task_run_config(_BASELINE, "Isaac-Cartpole-v0")
        cmd = gate._benchmark_cmd("Isaac-Cartpole-v0", cfg, Path("/tmp/out"))
        self.assertIn("--task", cmd)
        self.assertEqual(cmd[cmd.index("--task") + 1], "Isaac-Cartpole-v0")
        self.assertEqual(cmd[cmd.index("--num_frames") + 1], "300")
        self.assertEqual(cmd[cmd.index("--seed") + 1], "42")
        self.assertEqual(cmd[cmd.index("--num_envs") + 1], "4096")
        self.assertEqual(cmd[cmd.index("--benchmark_backend") + 1], "json")
        self.assertEqual(cmd[cmd.index("--output_path") + 1], "/tmp/out")
        self.assertIn("--headless", cmd)

    def test_variant_launches_real_gym_id_with_benchmark_args(self):
        cfg = gate._task_run_config(_BASELINE, "Isaac-Cartpole-v0@newton")
        cmd = gate._benchmark_cmd("Isaac-Cartpole-v0@newton", cfg, Path("/tmp/out"))
        self.assertEqual(cmd[cmd.index("--task") + 1], "Isaac-Cartpole-v0")
        self.assertIn("presets=newton", cmd)

    def test_num_envs_omitted_when_absent(self):
        cfg = {"task_id": "T", "num_envs": None, "num_frames": 10, "seed": 1, "benchmark_args": []}
        cmd = gate._benchmark_cmd("T", cfg, Path("/tmp/out"))
        self.assertNotIn("--num_envs", cmd)


class CacheEnv(unittest.TestCase):
    def test_none_when_no_cache_dir(self):
        self.assertIsNone(gate._cache_env(None))
        self.assertIsNone(gate._cache_env(""))

    def test_sets_warp_and_cuda_paths_and_creates_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = gate._cache_env(tmp)
            self.assertIsNotNone(env)
            assert env is not None  # narrow for type-checkers
            warp = Path(env["WARP_CACHE_PATH"])
            cuda = Path(env["CUDA_CACHE_PATH"])
            self.assertTrue(warp.is_dir())
            self.assertTrue(cuda.is_dir())
            # The overlay preserves the ambient environment.
            self.assertIn("PATH", env)


class _FakeProc:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class RunComparator(unittest.TestCase):
    def test_parses_result_token_over_exit_code(self):
        # WARN exits 0 but must be distinguished from a clean PASS.
        fake = _FakeProc(0, stdout="RESULT=WARN task=T delta_pct=-4.0\n")
        with mock.patch.object(gate.subprocess, "run", return_value=fake):
            with redirect_stdout(io.StringIO()):
                code, label = gate._run_comparator("T", Path("/tmp"), Path("/tmp/baseline.json"), None)
        self.assertEqual(code, 0)
        self.assertEqual(label, "WARN")

    def test_falls_back_to_verdict_name_without_result_token(self):
        fake = _FakeProc(gate.EXIT_HARD_FAILURE, stderr="boom, no token here\n")
        with mock.patch.object(gate.subprocess, "run", return_value=fake):
            with redirect_stdout(io.StringIO()):
                code, label = gate._run_comparator("T", Path("/tmp"), Path("/tmp/baseline.json"), None)
        self.assertEqual(code, gate.EXIT_HARD_FAILURE)
        self.assertEqual(label, "HARD_FAILURE")


class MainAggregation(unittest.TestCase):
    """Exercise ``main`` end-to-end in ``--dry-run`` (no GPU, no subprocess)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.baseline_path = Path(self._tmp.name) / "baseline.json"
        self.baseline_path.write_text(json.dumps(_BASELINE), encoding="utf-8")
        self.out_dir = Path(self._tmp.name) / "out"

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, tasks: list[str]) -> int:
        argv = [
            "--tasks", *tasks,
            "--baseline", str(self.baseline_path),
            "--output-dir", str(self.out_dir),
            "--dry-run",
        ]
        with redirect_stdout(io.StringIO()):
            return gate.main(argv)

    def test_all_valid_tasks_pass(self):
        self.assertEqual(self._run(["Isaac-Cartpole-v0"]), gate.EXIT_PASS)

    def test_missing_task_is_hard_failure(self):
        self.assertEqual(self._run(["Nope"]), gate.EXIT_HARD_FAILURE)

    def test_worst_verdict_wins_in_mixed_run(self):
        # One valid (PASS) + one missing (HARD_FAILURE) -> overall HARD_FAILURE.
        self.assertEqual(self._run(["Isaac-Cartpole-v0", "Nope"]), gate.EXIT_HARD_FAILURE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
