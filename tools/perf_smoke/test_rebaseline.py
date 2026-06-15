# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ``rebaseline.py`` (the variance study / rolling re-baseline tool).

Covers the pure-logic and store-level surface that needs no GPU: window stats,
the window-stats -> baseline-field mapping, current-baseline lookup, the rolling
window writer (flat vs env-fingerprint bucket, provenance stamping, cap), and the
boiling-frog guard end-to-end via ``main --from-stats --apply``.

Stdlib ``unittest``; also collectable by pytest via this directory's ``pytest.ini``.
Run directly::

    python3 tools/perf_smoke/test_rebaseline.py
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

import rebaseline  # noqa: E402


class WindowStats(unittest.TestCase):
    def test_none_for_empty_window(self):
        self.assertIsNone(rebaseline._window_stats([]))

    def test_robust_summary_of_known_window(self):
        stats = rebaseline._window_stats([100.0, 110.0, 90.0])
        assert stats is not None
        self.assertEqual(stats["n"], 3)
        self.assertEqual(stats["median"], 100.0)
        self.assertEqual(stats["min"], 90.0)
        self.assertEqual(stats["max"], 110.0)
        self.assertGreater(stats["cv_pct"], 0.0)

    def test_single_sample_reports_zero_spread(self):
        stats = rebaseline._window_stats([1234.0])
        assert stats is not None
        self.assertEqual(stats["n"], 1)
        self.assertEqual(stats["cv_pct"], 0.0)
        self.assertEqual(stats["mad"], 0.0)


class ProposedEntry(unittest.TestCase):
    def test_thresholds_hit_floor_for_low_variance(self):
        prop = rebaseline._proposed_entry({"cv_pct": 0.0, "median": 100000.0, "n": 5})
        self.assertEqual(prop["baseline_fps"], 100000.0)
        self.assertEqual(prop["warn_pct"], 5.0)
        self.assertEqual(prop["max_regression_pct"], 10.0)
        self.assertEqual(prop["n_runs"], 5)

    def test_thresholds_scale_with_variance(self):
        prop = rebaseline._proposed_entry({"cv_pct": 2.0, "median": 100.0, "n": 3})
        self.assertEqual(prop["warn_pct"], 6.0)  # max(3*cv, 5)
        self.assertEqual(prop["max_regression_pct"], 12.0)  # max(6*cv, 10)


class CurrentBaselineFps(unittest.TestCase):
    _BASELINE = {"T": {"per_gpu": {"NVIDIA L40S": {"baseline_fps": 4242.0}}}}

    def test_reads_nested_per_gpu_value(self):
        self.assertEqual(rebaseline._current_baseline_fps(self._BASELINE, "T", "NVIDIA L40S"), 4242.0)

    def test_none_when_task_or_gpu_absent(self):
        self.assertIsNone(rebaseline._current_baseline_fps(self._BASELINE, "T", "NVIDIA A100"))
        self.assertIsNone(rebaseline._current_baseline_fps(self._BASELINE, "Other", "NVIDIA L40S"))


class AppendWindow(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.history = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_flat_write_when_no_provenance(self):
        stats = {"samples": [100.0, 101.0], "walls": [10.0, 11.0]}
        n = rebaseline._append_window(self.history, "Isaac-Cartpole-v0", "NVIDIA L40S", stats, cap=20)
        self.assertEqual(n, 2)
        path = self.history / "Isaac-Cartpole-v0__NVIDIA_L40S.json"
        self.assertTrue(path.exists())
        store = json.loads(path.read_text())
        self.assertNotIn("fingerprint", store)
        self.assertEqual(store["samples"][0]["fps"], 100.0)
        self.assertEqual(store["samples"][0]["source"], "rebaseline")

    def test_bucketed_write_and_provenance_stamp(self):
        stats = {
            "samples": [100.0, 101.0],
            "walls": [10.0, 11.0],
            "provenance": {
                "fingerprint": "env-deadbeef0001",
                "commit": "abc1234",
                "warp": "1.13.0",
                "isaaclab": "6.6.1",
                "cuda": "12.4",
            },
        }
        rebaseline._append_window(self.history, "Isaac-Cartpole-v0", "NVIDIA L40S", stats, cap=20)
        path = self.history / "env-deadbeef0001" / "Isaac-Cartpole-v0__NVIDIA_L40S.json"
        self.assertTrue(path.exists(), "bucketed history file should be created under the fingerprint dir")
        store = json.loads(path.read_text())
        self.assertEqual(store["fingerprint"], "env-deadbeef0001")
        sample = store["samples"][0]
        self.assertEqual(sample["commit"], "abc1234")
        self.assertEqual(sample["warp"], "1.13.0")
        self.assertEqual(sample["isaaclab"], "6.6.1")
        self.assertEqual(sample["cuda"], "12.4")

    def test_appends_accumulate_and_prune_to_cap(self):
        first = {"samples": [1.0, 2.0], "walls": [1.0, 1.0]}
        rebaseline._append_window(self.history, "T", "G", first, cap=3)
        second = {"samples": [3.0, 4.0], "walls": [1.0, 1.0]}
        n = rebaseline._append_window(self.history, "T", "G", second, cap=3)
        self.assertEqual(n, 3)
        store = json.loads((self.history / "T__G.json").read_text())
        self.assertEqual([s["fps"] for s in store["samples"]], [2.0, 3.0, 4.0])


class BoilingFrogGuard(unittest.TestCase):
    """``main --from-stats --apply``: small drops applied, large drops refused."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.baseline_path = root / "baseline.json"
        self.history = root / "perf_history"
        self.stats_path = root / "stats.json"
        self.out_dir = root / "out"
        baseline = {
            "TaskNormal": {"num_envs": 1, "per_gpu": {"NVIDIA L40S": {"baseline_fps": 100000.0}}},
            "TaskRegress": {"num_envs": 1, "per_gpu": {"NVIDIA L40S": {"baseline_fps": 100000.0}}},
        }
        self.baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
        cached = {
            # ~1% drop -> within soft band -> applied.
            "TaskNormal": {
                "n": 3, "median": 99000.0, "mean": 99000.0, "cv_pct": 0.2, "mad": 50.0,
                "min": 98900.0, "max": 99100.0,
                "samples": [99000.0, 99050.0, 98950.0], "walls": [10.0, 10.0, 10.0],
            },
            # 20% drop -> beyond hard-drop -> refused (old value kept, no window write).
            "TaskRegress": {
                "n": 1, "median": 80000.0, "mean": 80000.0, "cv_pct": 0.0, "mad": 0.0,
                "min": 80000.0, "max": 80000.0,
                "samples": [80000.0], "walls": [10.0],
            },
        }
        self.stats_path.write_text(json.dumps(cached), encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def _run_apply(self) -> int:
        argv = [
            "--baseline", str(self.baseline_path),
            "--history-dir", str(self.history),
            "--output-dir", str(self.out_dir),
            "--from-stats", str(self.stats_path),
            "--apply",
        ]
        with redirect_stdout(io.StringIO()):
            return rebaseline.main(argv)

    def test_small_drop_applied_large_drop_refused(self):
        self.assertEqual(self._run_apply(), 0)
        baseline = json.loads(self.baseline_path.read_text())
        normal = baseline["TaskNormal"]["per_gpu"]["NVIDIA L40S"]["baseline_fps"]
        regress = baseline["TaskRegress"]["per_gpu"]["NVIDIA L40S"]["baseline_fps"]
        self.assertEqual(normal, 99000.0, "small drop should be written into baseline.json")
        self.assertEqual(regress, 100000.0, "hard-limit drop should keep the old baseline value")

    def test_refused_task_window_is_not_written(self):
        self._run_apply()
        self.assertTrue((self.history / "TaskNormal__NVIDIA_L40S.json").exists())
        self.assertFalse(
            (self.history / "TaskRegress__NVIDIA_L40S.json").exists(),
            "refused task must not contribute to the rolling window",
        )

    def test_force_applies_even_hard_drop(self):
        with redirect_stdout(io.StringIO()):
            rc = rebaseline.main([
                "--baseline", str(self.baseline_path),
                "--history-dir", str(self.history),
                "--output-dir", str(self.out_dir),
                "--from-stats", str(self.stats_path),
                "--apply", "--force",
            ])
        self.assertEqual(rc, 0)
        baseline = json.loads(self.baseline_path.read_text())
        self.assertEqual(baseline["TaskRegress"]["per_gpu"]["NVIDIA L40S"]["baseline_fps"], 80000.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
