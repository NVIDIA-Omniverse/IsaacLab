# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the env-fingerprint + per-sample provenance helpers in
``check_perf_regression.py`` and the bucketed history lookup.

These back the rolling-window bucketing: a run's environment (warp / isaaclab /
cuda) is hashed into a stable bucket key so incomparable software stacks never
share a baseline, and each stored sample is stamped with commit + versions for
auditability. The comparator reads the bucketed file in preference to the flat
("default") one.

Stdlib ``unittest``; also collectable by pytest via this directory's ``pytest.ini``.
Run directly::

    python3 tools/perf_smoke/test_history_fingerprint.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_perf_regression as cpr  # noqa: E402


def _result(warp="1.13.0", isaaclab="6.6.1", cuda="12.4", commit="abc1234"):
    """Minimal normalized-result shape carrying version + hardware provenance."""
    version: dict = {}
    if warp is not None:
        version["warp_version"] = warp
    if isaaclab is not None:
        version["isaaclab_version"] = isaaclab
    if commit is not None:
        version["dev"] = {"commit_hash": commit}
    hw: dict = {}
    if cuda is not None:
        hw["cuda_version"] = cuda
    out: dict = {}
    if version:
        out["version_info"] = version
    if hw:
        out["hardware_info"] = hw
    return out


class EnvFingerprint(unittest.TestCase):
    def test_deterministic_and_prefixed(self):
        fp1 = cpr.env_fingerprint(_result())
        fp2 = cpr.env_fingerprint(_result())
        self.assertEqual(fp1, fp2)
        assert fp1 is not None
        self.assertTrue(fp1.startswith("env-"))
        self.assertEqual(len(fp1), len("env-") + 12)

    def test_changes_with_environment(self):
        base = cpr.env_fingerprint(_result(warp="1.13.0"))
        bumped = cpr.env_fingerprint(_result(warp="1.14.0"))
        self.assertNotEqual(base, bumped)

    def test_commit_does_not_affect_fingerprint(self):
        # The fingerprint is the *environment*, not the code under test.
        a = cpr.env_fingerprint(_result(commit="aaaa"))
        b = cpr.env_fingerprint(_result(commit="bbbb"))
        self.assertEqual(a, b)

    def test_none_without_provenance(self):
        self.assertIsNone(cpr.env_fingerprint({}))
        self.assertIsNone(cpr.env_fingerprint(_result(warp=None, isaaclab=None, cuda=None)))


class ExtractCommit(unittest.TestCase):
    def test_reads_dev_commit_hash(self):
        self.assertEqual(cpr._extract_commit(_result(commit="deadbee")), "deadbee")

    def test_fallback_to_top_level_commit(self):
        self.assertEqual(cpr._extract_commit({"version_info": {"commit_hash": "top123"}}), "top123")

    def test_none_when_absent(self):
        self.assertIsNone(cpr._extract_commit({}))
        self.assertIsNone(cpr._extract_commit(_result(commit=None)))


class SampleProvenance(unittest.TestCase):
    def test_bundles_versions_commit_and_fingerprint(self):
        prov = cpr.sample_provenance(_result())
        self.assertEqual(prov["warp"], "1.13.0")
        self.assertEqual(prov["isaaclab"], "6.6.1")
        self.assertEqual(prov["cuda"], "12.4")
        self.assertEqual(prov["commit"], "abc1234")
        self.assertEqual(prov["fingerprint"], cpr.env_fingerprint(_result()))

    def test_omits_missing_fields(self):
        prov = cpr.sample_provenance(_result(warp=None, cuda=None, commit=None))
        self.assertNotIn("warp", prov)
        self.assertNotIn("cuda", prov)
        self.assertNotIn("commit", prov)
        self.assertEqual(prov["isaaclab"], "6.6.1")


class HistoryBasename(unittest.TestCase):
    def test_sanitizes_slashes_and_spaces(self):
        self.assertEqual(cpr.history_basename("Isaac-Cartpole-v0", "NVIDIA L40S"), "Isaac-Cartpole-v0__NVIDIA_L40S")
        self.assertEqual(cpr.history_basename("a/b", "g g"), "a_b__g_g")


class HistoryWindowLookup(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.history = Path(self._tmp.name)
        self.fp = "env-deadbeef0001"
        self.task = "Isaac-Cartpole-v0"
        self.gpu = "NVIDIA L40S"
        safe = cpr.history_basename(self.task, self.gpu)
        self.flat = self.history / f"{safe}.json"
        self.bucket = self.history / self.fp / f"{safe}.json"
        self.bucket.parent.mkdir(parents=True, exist_ok=True)
        self.flat.write_text(json.dumps({"samples": [{"fps": 1.0}], "marker": "flat"}), encoding="utf-8")
        self.bucket.write_text(json.dumps({"samples": [{"fps": 2.0}], "marker": "bucket"}), encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def test_bucket_preferred_when_fingerprint_given(self):
        window = cpr._history_window(str(self.history), self.fp, self.task, self.gpu)
        self.assertEqual(window["marker"], "bucket")

    def test_flat_fallback_when_bucket_missing(self):
        window = cpr._history_window(str(self.history), "env-nonexistent", self.task, self.gpu)
        self.assertEqual(window["marker"], "flat")

    def test_flat_used_when_no_fingerprint(self):
        window = cpr._history_window(str(self.history), None, self.task, self.gpu)
        self.assertEqual(window["marker"], "flat")

    def test_empty_when_nothing_exists(self):
        window = cpr._history_window(str(self.history), None, "Other-Task", self.gpu)
        self.assertEqual(window, {})

    def test_empty_when_history_dir_disabled(self):
        self.assertEqual(cpr._history_window(None, self.fp, self.task, self.gpu), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
