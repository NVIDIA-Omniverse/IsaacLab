# Bisection Agent Overview

## TL;DR

- The **perf gate** (Phase 1) tells us *that* a benchmark regressed on a PR or
branch.
- The **bisection agent** (Phase 2) tells us *which commit* caused it, then writes
a first-pass explanation of what that commit changed.
- It is **local-first**: engineers should be able to run it from their own
development environment before escalating anywhere else.
- It is **artifact-driven**: every stage reads and writes plain JSON files, so it
can be inspected, rerun, cached, or called from another tool.
- It reuses perf-smoke pieces where possible: the benchmark execution path and
the `perf_smoke_test_result.json` result contract stay shared.

## The Problem

The perf gate answers one question:

> Did this change make the benchmark slower?

That is useful, but when the answer is "yes," an engineer still has to manually
`git bisect` across dozens of commits, running an expensive GPU benchmark at each
step, to find the culprit. That is slow and easy to get wrong because of
benchmark noise, environment differences, and infrastructure failures.

The bisection agent automates exactly that loop:

> Given a known-good commit and a known-bad commit, find the **first** commit
> where the benchmark went bad, and summarize what it changed.

## How It Works

The local-first flow starts by measuring the known-good and known-bad refs in
the same local environment. If the bad ref is not meaningfully slower locally,
the agent stops instead of pretending it found a culprit. If the signal
reproduces, the agent runs a binary search over the commit range. At each
midpoint commit it runs one benchmark cell and compares the result to the local
good/bad measurements.

```text
perf gate or engineer identifies a regression
  -> plan.json describes what to bisect (range, task, backend, runner)
  -> harness lists commits between known-good and known-bad
  -> local preflight measures known-good and known-bad refs
  -> for each midpoint commit:
        single-commit runner checks out that commit in isolation,
        runs ONE benchmark cell, emits normal perf-smoke artifacts
     paired-reference comparator labels it GOOD / BAD / UNCLEAR
  -> engine narrows the range to the first BAD commit
  -> summary.json reports the first bad commit
  -> diagnosis.md explains what changed
```

The key idea: the bisection logic does not run IsaacLab directly. It delegates
one candidate measurement at a time to the same single-commit runner and result
format used by the smoke test tooling, then classifies the artifact with the
selected comparison mode.

## Architecture At A Glance

The bisection agent is not one monolithic script. It is a small set of files that
each own one part of the workflow:

- `bisection_plan_from_gate.py` creates a `plan.json` from a perf-smoke failure.
- `bisection_harness.py` is the command-line entry point that starts the run.
- `bisection/engine.py` performs the binary search over commits.
- `bisect_single_commit_runner.py` checks out one candidate commit and runs one
  benchmark cell.
- `bisection/paired_reference.py` compares candidates against locally measured
  good/bad refs.
- `bisection/oracle_adapter.py` keeps the older historical-baseline comparison
  path available for manual/optional flows.
- `diagnose_bad_commit.py` writes the first-pass diagnosis after the first bad
  commit is found.

The most important design choice is that **search**, **execution**, **verdicts**,
and **diagnosis** are separate. The engine only knows how to search commits. The
single-commit runner knows how to run IsaacLab for one candidate. The oracle
adapter owns the bridge back to the perf gate. The diagnosis step is intentionally
separate so deeper profiling can be added later without changing the core
bisection loop.

The pieces communicate through artifacts instead of hidden in-memory state:
`plan.json`, `status.json`, `results/*.json`, `summary.json`, and
`diagnosis.md`. That makes the agent easier to debug, rerun, cache, and call
from another tool.

## The Pieces


| Component                | File                                               | Job                                                                                                                       |
| ------------------------ | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Plan creator**         | `bisection_plan_from_gate.py`                      | Turns a gate artifact ("we found a regression") into a `plan.json` bisection job. The bridge between Phase 1 and Phase 2. |
| **Harness**              | `bisection_harness.py`                             | CLI entry point and coordinator.                                                                                          |
| **Engine**               | `bisection/engine.py`                              | The binary-search algorithm, timeouts, and retries.                                                                       |
| **Single-commit runner** | `bisect_single_commit_runner.py`                   | Runs exactly one task/backend at exactly one commit and emits normal perf-smoke artifacts.                                 |
| **Paired comparator**    | `bisection/paired_reference.py`                    | Labels local candidates GOOD/BAD/UNCLEAR against locally measured good/bad refs.                                           |
| **Oracle adapter**       | `bisection/oracle_adapter.py`                      | Keeps the historical-baseline comparison path available for optional/manual runs.                                           |
| **Diagnosis**            | `diagnose_bad_commit.py`, `bisection/diagnosis.py` | Compares the first bad commit to its parent: changed files, subsystem guess, FPS delta.                                   |


### Runner Modes

The single-commit runner supports three modes so the same logic works for local
development and optional manual escalation:

- `synthetic`: fake FPS, for fast control-flow demos without a GPU.
- `local-source`: real GPU run using an isolated checkout plus the host IsaacLab
environment.
- `docker-source`: real run inside a fixed perf-smoke container.

## The Data Contract (`plan.json`)

Everything the harness needs to run a bisection lives in one structured file
(schema v2):

```json
{
  "schema_version": 2,
  "task_id": "Isaac-Cartpole-Direct",
  "backend_key": "physx",
  "good_ref": "<known-good SHA>",
  "bad_ref": "<known-bad SHA>",
  "runner": {
    "mode": "local-source",
    "source_dir": "{output_dir}/sources/{commit_sha}",
    "jit_cache": "{output_dir}/jit-cache",
    "kit_cache": "{output_dir}/kit-cache"
  },
  "measurement": {
    "reference_runs": 3,
    "max_reference_runs": 7,
    "candidate_runs": 1,
    "max_candidate_runs": 3,
    "min_regression_pct": 5.0,
    "reference_noise_multiplier": 2.0
  },
  "timeout": { "candidate_timeout_s": 900 },
  "retry": { "max_attempts": 2, "retry_delay_s": 10 }
}
```

Per-candidate evidence is also recorded:

- one artifact directory per attempt,
- command exit code,
- duration,
- timeout status,
- retry reason,
- final artifact directory.

That makes a flaky or failed candidate easier to debug after the fact.

## Local Comparison Model

The local-first path does not use stored gate baselines as numeric FPS targets.
Those baselines were measured under a specific gate environment, while local
bisection runs in the engineer's local managed environment. Instead, the agent
uses paired local references:

1. Measure `good_ref` locally.
2. Measure `bad_ref` locally.
3. Confirm `bad_ref` is meaningfully slower under the same local setup.
4. Compare midpoint candidates against the local `good_ref` signal.

The statistical model follows the same robustness goals as the perf gate, but it
is not the exact same oracle. The gate uses historical rolling-window baselines
with median/MAD-style thresholds. Local bisection has no trusted historical
baseline, so it uses repeated local reference measurements:

- reference measurements start at `reference_runs` and can grow up to
  `max_reference_runs` when the signal is noisy,
- FPS is summarized with medians,
- local reference noise is estimated with a robust median-absolute-deviation
  spread,
- the effective regression threshold is
  `max(min_regression_pct, reference_noise_multiplier * reference_noise_pct)`,
- candidates near the threshold are labeled `UNCLEAR` instead of being forced
  into GOOD or BAD.

This keeps the local-first path from depending on arbitrary one-off FPS samples
while preserving a separate gate-faithful comparison path for optional/manual
flows.

## What's Been Proven

**1. Control flow (synthetic).** Plan -> bisection -> classification -> summary
-> diagnosis all work without a GPU.

**2. Real regression on a real GPU.** We created a demo branch
(`neilm/bisection-real-demo-range`) that deliberately slows down the Cartpole
env:

```text
source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_direct_env.py
```

The perf gate flagged it as a real regression:

- task/backend: `Isaac-Cartpole-Direct/physx`
- verdict: `BLOCK`, about `-25%` (around 281k FPS baseline to around 211k FPS)

The harness then bisected real commits and correctly identified the first bad
commit (`af9729fe56dbe256aa2a255411fe82814cb9bb50`), and diagnosis pointed at the
Cartpole env file.

**3. Local-first direction.** The current work is moving the default workflow to
paired local measurements: measure good/bad refs locally, confirm the signal
reproduces, then bisect from that local signal. The manual workflow remains
available as an optional path, but it is not the normal developer entry point.

## How To Run It

### Locally: Paired-Reference Bisection

```bash
./isaaclab.sh -p tools/perf_smoke_test/bisection_harness.py run-local \
  --good_ref <known-good-SHA> \
  --bad_ref <known-bad-SHA> \
  --task_id Isaac-Cartpole-Direct \
  --backend_key physx \
  --work_dir perf-output/bisection-local
```

Use `--runner_mode synthetic` for a GPU-free control-flow check, or
`--runner_mode local-source` for a real local benchmark run.

### Optional Manual Workflow

The workflow is `workflow_dispatch`-only and is kept as a manual path. Dispatch
it against the branch carrying the tooling when you explicitly want that path:

```bash
gh workflow run perf-smoke-bisect.yaml --ref neilm/perf-smoke-bisect \
  -f good_ref=<parent-of-regression-SHA> \
  -f bad_ref=<bad-tip-SHA> \
  -f task_id=Isaac-Cartpole-Direct \
  -f backend_key=physx
```

It uploads `summary.json`, per-commit `results/*.json`, and `diagnosis.md` as run
artifacts.

## Current Limitations

- `local-source` depends on host-specific IsaacLab setup.
- Local paired-reference bisection assumes the regression reproduces locally.
- Reference measurement is adaptive: the agent starts with `reference_runs`, adds
samples up to `max_reference_runs` when measurements are noisy, and only proceeds
once the local signal is stable enough.
- The local threshold is also noise-aware: the configured `min_regression_pct` is
a floor, and the effective threshold rises when repeated good/bad measurements
are noisy.
- Timeout/retry exists, but the policy is still simple.
- **Diagnosis is a git diff plus benchmark context today.** It identifies the commit and changed files but does not yet prove root cause via Nsight or GPU traces.
- It bisects one regressed task/backend cell at a time.
- Gradual regressions accumulated across many commits are out of scope for this
first version.

## Where It's Headed

1. **Make local paired-reference bisection the default path** for engineers.
2. **Tune measurement policy** so noisy local measurements become UNCLEAR, not a
  false BAD.
3. **Add deeper diagnosis** by optionally capturing Nsight Systems traces for the
  parent and first-bad commit and attaching the comparison to `diagnosis.json`.
4. **Improve environment handling** after the local-first MVP is proven, without
  moving environment decisions into the single-commit runner.
