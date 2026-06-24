# Bisection Agent Overview

## TL;DR

- The **perf gate** (Phase 1) tells us *that* a benchmark regressed on a PR or
branch.
- The **bisection agent** (Phase 2) tells us *which commit* caused it, then writes
a first-pass explanation of what that commit changed.
- It is **artifact-driven**: every stage reads and writes plain JSON files, so it
can be inspected, rerun, or launched from CI, Slack, or another agent.
- It **reuses the perf-smoke oracle** to label commits. It does not invent a
second regression detector.
- It is proven end-to-end on a real GPU with a real injected regression, and is
currently being validated on the RTX PRO 6000 CI fleet through Docker.

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

The agent runs a binary search over the commit range. At each midpoint commit it
runs one real benchmark and asks the perf-smoke oracle to classify the result.

```text
perf gate finds a regression
  -> plan.json describes what to bisect (range, task, backend, baselines, runner)
  -> harness lists commits between known-good and known-bad
  -> for each midpoint commit:
        single-commit runner checks out that commit in isolation,
        runs ONE benchmark cell, emits normal perf-smoke artifacts
     oracle adapter labels it GOOD / BAD / SKIP
  -> engine narrows the range to the first BAD commit
  -> summary.json reports the first bad commit
  -> diagnosis.md explains what changed
```

The key idea: the bisection logic never measures performance itself. It runs the
same benchmark and same oracle the gate already uses, so a "BAD" verdict during
bisection means the same thing as a "BAD" verdict on a PR.

## Architecture At A Glance

The bisection agent is not one monolithic script. It is a small set of files that
each own one part of the workflow:

- `bisection_plan_from_gate.py` creates a `plan.json` from a perf-smoke failure.
- `bisection_harness.py` is the command-line entry point that starts the run.
- `bisection/engine.py` performs the binary search over commits.
- `bisect_single_commit_runner.py` checks out one candidate commit and runs one
  benchmark cell.
- `bisection/oracle_adapter.py` asks the existing perf-smoke oracle whether that
  candidate is GOOD, BAD, or SKIP.
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
`diagnosis.md`. That makes the agent easier to debug, rerun, and eventually call
from CI, Slack, or another AI agent.

## The Pieces


| Component                | File                                               | Job                                                                                                                       |
| ------------------------ | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Plan creator**         | `bisection_plan_from_gate.py`                      | Turns a gate artifact ("we found a regression") into a `plan.json` bisection job. The bridge between Phase 1 and Phase 2. |
| **Harness**              | `bisection_harness.py`                             | CLI entry point and coordinator.                                                                                          |
| **Engine**               | `bisection/engine.py`                              | The binary-search algorithm, timeouts, and retries.                                                                       |
| **Single-commit runner** | `bisect_single_commit_runner.py`                   | Runs exactly one task/backend at exactly one commit and emits normal perf-smoke artifacts.                                 |
| **Oracle adapter**       | `bisection/oracle_adapter.py`                      | Reuses the perf-smoke oracle to label a commit GOOD/BAD/SKIP.                                                              |
| **Diagnosis**            | `diagnose_bad_commit.py`, `bisection/diagnosis.py` | Compares the first bad commit to its parent: changed files, subsystem guess, FPS delta.                                   |


### Runner Modes

The single-commit runner supports three modes so the same logic works from a
local demo up to GPU CI:

- `synthetic`: fake FPS, for fast control-flow demos without a GPU.
- `local-source`: real GPU run using an isolated checkout plus the host IsaacLab
environment.
- `docker-source`: real run inside the perf-smoke CI container, intended for the
GPU fleet.

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
    "mode": "docker-source",
    "image": "nvcr.io/nvidian/isaac-lab:latest-perf",
    "source_dir": "{output_dir}/candidate-source",
    "jit_cache": "{output_dir}/jit-cache",
    "kit_cache": "{output_dir}/kit-cache"
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

**3. Docker/CI on the fleet (in progress).** A manual workflow
(`.github/workflows/perf-smoke-bisect.yaml`) runs the whole flow in
`docker-source` mode on the RTX PRO 6000 fleet, pulling the perf image from NVCR.
Initial runs have proven checkout, NVCR image pull, GPU detection, and the real
known-good benchmark cell. The remaining validation is to complete the full
Docker-backed bisection loop on the fleet.

## How To Run It

### Locally (No GPU): See The Control Flow

```bash
./isaaclab.sh -p tools/perf_smoke_test/bisection_harness.py run \
  --plan <path-to-synthetic-plan.json> \
  --output_dir perf-output/bisection-demo
```

### On The GPU CI Fleet (Docker)

The workflow is `workflow_dispatch`-only. Dispatch it against the branch carrying
the tooling:

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

- Docker/CI validation is still being finalized on the fleet.
- `local-source` depends on host-specific setup; `docker-source` is the path to a
controlled, repeatable environment.
- Timeout/retry exists, but the policy is simple and should be tuned with real CI
data.
- **Diagnosis is a git diff plus benchmark context today.** It identifies the commit and changed files but does not yet prove root cause via Nsight or GPU traces.
- It bisects one regressed task/backend cell at a time.

## Where It's Headed

1. **Validate Docker mode** end-to-end on the fleet.
2. **Tune timeout/retry** so infrastructure flakiness becomes SKIP, not a false
  BAD.
3. **Add deeper diagnosis** by optionally capturing Nsight Systems traces for the
  parent and first-bad commit and attaching the comparison to `diagnosis.json`.
4. **Promote the workflow** to the standard entry point: gate finds a regression,
  maintainer launches bisection from the failed artifact, and a GPU runner
   reports the first bad commit plus diagnosis.
