# Performance Regression Smoke Gate

A small CI gate that catches **GPU performance regressions** in Isaac Lab before
they merge. On a pull request it runs a few short benchmark tasks on a fixed GPU,
compares the measured speed against a rolling baseline, and reports
PASS / WARN / BLOCK back onto the PR.

This document explains, in plain terms, what each file in this commit does.

## The idea in one paragraph

Every PR, we run a handful of cheap, stable Isaac Lab tasks (e.g. Cartpole) on a
known GPU and measure how fast they step (frames per second). We keep a short
history of recent "known-good" numbers per task and GPU. If a PR's measurement
drops well below that history, the gate flags it. Small wobble is expected, so a
mild dip is an advisory **WARN** and only a real, sustained drop is a **BLOCK**.

## What's in this directory

### The gate itself (the parts that do the work)

| File | Plain-English purpose |
|---|---|
| `run_perf_gate.py` | The orchestrator. For each task it launches the Isaac Lab benchmark as its own process, then hands the result to the comparator and prints the verdict. |
| `check_perf_regression.py` | The comparator (pure logic, no GPU needed). Reads a benchmark result + the baseline/history and decides PASS, WARN, or BLOCK using a median + MAD test. |
| `baseline.json` | Per-task run settings (num envs, frames, seed) **and** a static fallback speed used when there isn't enough history yet. Keyed by task and GPU. |
| `baseline_overrides.json` | Manual, in-tree threshold overrides that ride along with a PR when a one-off adjustment is needed. |
| `perf_history/` | The rolling window of recent measurements, one JSON per task+GPU. This is what the comparator normally judges against. |

### Helper tools

| File | Plain-English purpose |
|---|---|
| `rebaseline.py` | Re-computes the baseline/history from fresh runs (used when moving to new hardware). |
| `seed_history.py` | Seeds the `perf_history/` window from existing runs. |
| `demo_regression.py` | Injects a fake slowdown so you can watch the gate correctly BLOCK. |
| `warp_replicator_shim.py` | Small compatibility fix so the camera/RTX tasks import under newer Warp. Needed on arm64. |

### Tests

| File | Plain-English purpose |
|---|---|
| `test_check_perf_regression.py` | Unit tests for the comparator logic (no GPU). |
| `test_stress_check_perf_regression.py` | Heavier stress/edge-case tests for the comparator. |
| `test_perf_gate.py` | The pytest entry point CI uses to drive a single task end-to-end. |
| `pytest.ini` | Local pytest config for this directory. |

### CI wiring (in `.github/`)

| File | Plain-English purpose |
|---|---|
| `workflows/perf-gate.yml` | The GitHub Actions workflow. Runs the gate on the arm64 L40S fleet, one job per task, and posts a status back to the PR. |
| `workflows/perf-rebaseline.yml` | A manual workflow to re-bless baselines on the runner. |
| `copy-pr-bot.yaml` | Enables NVIDIA's copy-pr-bot so PRs are mirrored to `pull-request/*` branches (required for the self-hosted fleet). |

## The verdict model

| Verdict | Meaning | Effect |
|---|---|---|
| **PASS** | Within or above expected speed. | Gate succeeds. |
| **WARN** | Mild dip, within noise. | Advisory only; does not fail. |
| **BLOCK** | A real regression, or a structural problem (missing/blank result, unknown GPU, config mismatch). | Gate fails for that task. |

## Running it locally

Run the full gate for one task (needs a GPU + Isaac Lab installed):

```bash
./isaaclab.sh -p tools/perf_smoke/run_perf_gate.py --tasks Isaac-Cartpole-v0
```

Run just the comparator tests (no GPU needed):

```bash
python3 tools/perf_smoke/test_check_perf_regression.py
```

## Current status

- Target hardware: **arm64 L40S** on NVIDIA's shared self-hosted fleet.
- The gate is **advisory** to start (`continue-on-error: true`): it reports a
  verdict on the PR but does not block merges yet. It is flipped to required
  once cross-runner variance is confirmed.
- Baselines must be **re-blessed on the runner fleet** before the verdict is
  authoritative; the values committed here are calibration starting points.
