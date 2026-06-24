# Bisection Agent Architecture Overview

This is a short, plain-language overview of the bisection agent POC: what it is,
how it works, what we have proven, and what needs to improve next.

## One-Sentence Summary

The bisection agent starts from a performance regression found by the perf gate
and automatically searches the commit history to find the first commit that made
the benchmark go bad.

## Problem It Solves

Phase 1, the perf gate, answers:

> Did this PR or branch regress performance?

Phase 2, the bisection agent, answers:

> Which commit introduced that regression?

The agent does not replace the perf gate. It builds on top of it. The perf gate
detects the regression, and the bisection agent uses that result as the starting
point for deeper investigation.

## Core Design Idea

The main design choice is to keep the bisection agent artifact-driven.

Instead of hiding state inside one long script, each stage writes simple files:

- `plan.json`: what regression to investigate and how to run candidates
- `status.json`: current progress while the search is running
- `results/*.json`: one result per tested commit
- `summary.json`: final bisection answer
- `diagnosis.json` and `diagnosis.md`: first-pass explanation of the bad commit

This makes the harness easy to inspect, debug, rerun, or eventually launch from
CI, Slack, or another agent.

## High-Level Flow

1. The perf gate runs a benchmark and produces normal gate artifacts.
2. The bisection plan creator scans those artifacts and finds a regressed
   task/backend cell.
3. It writes `plan.json`, which describes the commit range, task/backend, GPU,
   baselines, gate config, and runner command.
4. The bisection harness reads `plan.json` and builds the list of commits between
   a known-good ref and known-bad ref.
5. The harness tests midpoint commits, like a normal binary search.
6. For each candidate commit, the single-commit runner checks out that source in
   isolation, runs one benchmark cell, and emits normal perf-gate artifacts.
7. The oracle adapter reuses the existing perf-gate oracle to classify that
   candidate as `GOOD`, `BAD`, or `SKIP`.
8. The bisection engine narrows the search until it finds the first bad commit.
9. The diagnosis step compares the first bad commit to its parent and writes a
   short explanation.

In simple terms:

```text
perf gate finds a regression
-> plan.json describes what to bisect
-> harness tests commits
-> oracle labels each commit GOOD/BAD/SKIP
-> summary.json reports the first bad commit
-> diagnosis.md explains what changed
```

## Main Components

### Plan Creator

File:

```text
tools/perf_regression_gate/bisection_plan_from_gate.py
```

This is the bridge between Phase 1 and Phase 2.

It reads perf-gate artifacts, re-evaluates them with the existing oracle, keeps
the cells that are bad enough to bisect, and writes `plan.json`.

In plain terms:

> The plan creator turns "the gate found a regression" into "here is a bisection
> job to run."

### Bisection Harness And Engine

Files:

```text
tools/perf_regression_gate/bisection_harness.py
tools/perf_regression_gate/bisection/engine.py
```

The harness is the command-line entry point. The engine is the binary-search
logic.

It resolves the good and bad refs, lists the commits between them, tests a
midpoint commit, and then decides whether to search earlier or later commits.

In plain terms:

> The harness is the coordinator, and the engine is the search algorithm.

### Single-Commit Runner

File:

```text
tools/perf_regression_gate/bisect_single_commit_runner.py
```

This runs exactly one benchmark task/backend at exactly one commit.

It currently supports three modes:

- `synthetic`: fake FPS values for fast control-flow demos
- `local-source`: real local GPU runs using an isolated checkout plus the host
  IsaacLab environment
- `docker-source`: real Docker-based runs intended for CI or GPU fleet usage

In plain terms:

> The single-commit runner answers "how fast is this one commit for this one
> benchmark?"

### Oracle Adapter

File:

```text
tools/perf_regression_gate/bisection/oracle_adapter.py
```

This lets the bisection agent reuse the Phase 1 perf-gate oracle.

It loads the candidate commit's normal benchmark artifacts and asks the oracle
whether that commit is `GOOD`, `BAD`, or `SKIP`.

In plain terms:

> The oracle adapter keeps the bisection agent from inventing a second regression
> detector.

### Diagnosis Step

Files:

```text
tools/perf_regression_gate/diagnose_bad_commit.py
tools/perf_regression_gate/bisection/diagnosis.py
```

This runs after the first bad commit is found.

Right now, diagnosis is intentionally simple. It compares the first bad commit to
its parent, lists changed files, classifies the changed subsystem based on file
paths, and reports FPS deltas.

In plain terms:

> Diagnosis currently says "this commit changed these files and coincides with
> this performance drop."

It is not yet a deep profiler. It does not yet prove the root cause from Nsight
or GPU traces.

## What We Proved

The POC now works in two levels.

First, the synthetic demo proved the control flow:

- generate a plan
- run bisection
- classify commits
- write summary and diagnosis artifacts

Second, the real L40S demo proved the same architecture with real IsaacLab
benchmark runs.

For the real demo, we created a short branch:

```text
neilm/bisection-real-demo-range
```

The branch intentionally introduced a slowdown in:

```text
source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_direct_env.py
```

The perf gate artifact reported a real regression:

- task/backend: `Isaac-Cartpole-Direct/physx`
- gate verdict: `BLOCK`
- regression: about `-25%`
- baseline: about `281k` FPS
- bad run: about `211k` FPS

The bisection harness then tested real commits and found the expected first bad
commit:

```text
af9729fe56dbe256aa2a255411fe82814cb9bb50
```

The diagnosis report correctly pointed to the Cartpole environment file.

## Hardening Added After The First Demo

The POC now has a stronger plan and execution contract than the original demo.

`plan.json` supports a structured v2 shape:

```json
{
  "schema_version": 2,
  "runner": {
    "mode": "local-source",
    "source_dir": "{output_dir}/candidate-source",
    "jit_cache": "{output_dir}/jit-cache",
    "kit_cache": "{output_dir}/kit-cache"
  },
  "timeout": {
    "candidate_timeout_s": 900
  },
  "retry": {
    "max_attempts": 2,
    "retry_delay_s": 10
  }
}
```

The old `runner_command` string still works for backwards compatibility, but new
plans should prefer the structured `runner`, `timeout`, and `retry` objects.

The harness now records richer per-candidate evidence:

- one artifact directory per attempt
- command exit code
- duration
- timeout status
- retry metadata
- final artifact directory

This means a failed or flaky candidate is easier to debug than before.

The structured plan was validated in two ways:

- synthetic bisection still works with the v2 plan shape
- real `local-source` bisection still finds
  `af9729fe56dbe256aa2a255411fe82814cb9bb50`

A manual Docker-backed workflow now exists at:

```text
.github/workflows/perf-gate-bisect.yaml
```

That workflow is intended to prove the next milestone: the same bisection flow
running through `docker-source` on the GPU CI fleet.

## Current Limitations

This is still a POC, not a production service.

The biggest limitations are:

- Docker/CI validation still needs to be run on a pushed branch with access to
  the GPU fleet and perf-gate CI image.
- `local-source` works on the local L40S host but depends on the host
  environment and host-specific setup.
- Retry behavior exists, but the policy is still simple and should be tuned as
  real infrastructure failures are observed.
- Diagnosis is mostly git-diff plus benchmark context; it does not yet include
  Nsight traces or deeper performance analysis.
- The current POC focuses on one regressed task/backend cell at a time.

## Next Improvements

The next work should make the POC more robust and easier to run repeatedly.

### 1. Validate Docker Mode

`local-source` is useful for local demos, but production CI should likely use
`docker-source` so the runtime environment is controlled and repeatable.

The next step is to run `.github/workflows/perf-gate-bisect.yaml` on a
Docker-enabled GPU runner and confirm it finds the same known bad commit.

### 2. Tune Timeout And Retry Policy

The harness can now time out and retry candidate runs, but the exact policy
should be tuned with real CI data.

The key distinction is:

- real performance `BAD`
- real performance `GOOD`
- infrastructure failure that should be retried
- persistent inconclusive result that should become `SKIP`

### 3. Add Deeper Diagnosis

The current diagnosis is a first-pass report. The next version should optionally
run parent vs first-bad profiling.

Possible additions:

- capture Nsight Systems traces for parent and first bad
- compare CPU/GPU timeline changes
- compare kernel time and synchronization stalls
- compare startup time vs runtime step time
- attach trace paths and profiler summaries to `diagnosis.json`

### 4. Promote The CI Workflow

After the manual workflow is validated, it can become the standard bisection
entry point:

1. A perf gate run finds a regression.
2. A maintainer launches bisection from the failed artifact.
3. The workflow runs on a GPU runner.
4. It uploads `summary.json`, `results/*.json`, and `diagnosis.md`.

## How To Explain It Out Loud

The simplest explanation is:

> We already have a perf gate that can tell us when a benchmark got slower. The
> bisection agent is the next layer: it takes that failed gate result, tests the
> commits between a known-good and known-bad point, and uses the same oracle as
> the gate to label each commit. Once it finds the first bad commit, it writes a
> short diagnosis report showing the changed files and the measured FPS drop.

And the current status is:

> We have proven the full loop with real IsaacLab runs on an L40S, then hardened
> the plan contract with structured runner settings, timeouts, and retries. The
> next big milestone is proving the same flow through Docker/CI; deeper profiling
> can come after the bisection infrastructure is reliable.
