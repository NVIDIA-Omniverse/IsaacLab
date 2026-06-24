# IsaacLab Bisection Agent Initial POC

This document describes the initial bisection-agent POC that sits after the
performance-regression gate.

The gate answers:

> Did this PR regress a benchmark cell?

The bisection POC answers:

> Given a regressed IsaacLab task/backend, which commit most likely introduced
> the regression, and what changed in that commit?

## What We Are Building

We are not replacing the AI agents people already use in their workflows. The POC
is a small IsaacLab-specific bisection harness that an existing agent, GitHub
workflow, Slack bot, or human can call.

The harness standardizes:

- how a bisection is planned
- how one candidate commit is evaluated
- how progress is tracked
- how the first bad commit is reported
- how the first bad commit is diagnosed

The harness deliberately keeps the phase-1 gate code unchanged. It consumes the
gate's existing artifact contract.

## Demo Flow

The demo is designed to work backward from what we want to show:

1. A gate run has a regressed benchmark cell.
2. The harness converts that gate output into a bisection plan.
3. The bisection engine tests candidate commits.
4. The existing perf-gate oracle labels each candidate `GOOD`, `BAD`, or `SKIP`.
5. The engine reports the suspected first bad commit.
6. The diagnosis step compares the first bad commit to its parent and writes a
   human-readable report.

The current demo is synthetic: it uses stub benchmark output instead of running
Isaac Sim. This proves the control flow without using GPU time. The next step is
to replace the synthetic runner with a real source-isolated Docker runner.

## Files Added

```text
tools/perf_regression_gate/
  bisection/
    __init__.py
    diagnosis.py
    engine.py
    gate_adapter.py
    git_utils.py
    io.py
    models.py
    oracle_adapter.py
  bisection_harness.py
  bisection_plan_from_gate.py
  bisect_single_commit_runner.py
  diagnose_bad_commit.py
  docs/bisection-agent-initial-poc.md
```

Existing gate files were not modified for this POC.

## Architecture

### 1. Gate Adapter

Entry point:

```bash
tools/perf_regression_gate/bisection_plan_from_gate.py
```

Role:

- scan a gate artifact directory
- evaluate each cell with the existing oracle
- find cells with `bisect_verdict == BAD`
- select one regressed task/backend
- write `gate_regressions.json`
- write `plan.json`

This is the bridge from phase 1 to phase 2.

### 2. Bisection Engine

Entry point:

```bash
tools/perf_regression_gate/bisection_harness.py
```

Core module:

```bash
tools/perf_regression_gate/bisection/engine.py
```

Role:

- resolve `good_ref` and `bad_ref`
- build the ordered candidate commit list
- choose midpoint commits
- call the configured single-commit runner
- read standard perf-gate artifacts
- classify candidate commits with the existing oracle
- write `status.json`, `results/*.json`, and `summary.json`

### 3. Single-Commit Runner

Entry point:

```bash
tools/perf_regression_gate/bisect_single_commit_runner.py
```

Current POC role:

- use `dev/stub_benchmark.py` to create normal perf-gate artifacts
- emit good FPS before a configured synthetic first-bad ref
- emit bad FPS at and after that first-bad ref
- optionally seed matching local baselines for the demo

Future real role:

- checkout one candidate commit in an isolated source tree
- run exactly one IsaacLab task/backend
- write the same artifact files
- keep the runtime/image/GPU/task config stable across candidates

### 4. Diagnosis

Entry point:

```bash
tools/perf_regression_gate/diagnose_bad_commit.py
```

Core module:

```bash
tools/perf_regression_gate/bisection/diagnosis.py
```

Role:

- read `summary.json`
- find the suspected first bad commit
- compare it to its parent
- summarize changed files and likely subsystem
- compare last-good vs first-bad metrics when available
- write `diagnosis.json`
- write `diagnosis.md`

This is intentionally lightweight. Nsight traces are a future diagnosis input,
not required for the initial POC.

## Artifact Contract

A bisection run directory looks like this:

```text
perf-output/bisection-poc/
  gate-artifacts/
    <task>/<backend>/
      launch_config.json
      benchmark.log
      perf_regression_gate_info.json
      perf_regression_gate_result.json
  plan/
    gate_regressions.json
    plan.json
  run/
    plan.resolved.json
    candidates.json
    status.json
    results/
      <sha>.json
    artifacts/
      <sha>/<task>/<backend>/
        launch_config.json
        benchmark.log
        perf_regression_gate_info.json
        perf_regression_gate_result.json
    summary.json
    diagnosis.json
    diagnosis.md
```

This mirrors YuTeh's useful `plan/status/results/summary` pattern while keeping
the evaluation IsaacLab-specific.

## Demo Commands

Start clean:

```bash
rm -rf perf-output/bisection-poc
mkdir -p perf-output/bisection-poc/gate-artifacts/Isaac-Cartpole-Direct/physx
```

Create a synthetic regressed gate artifact:

```bash
./isaaclab.sh -p tools/perf_regression_gate/bisect_single_commit_runner.py \
  --commit HEAD \
  --task_id Isaac-Cartpole-Direct \
  --backend_key physx \
  --artifact_dir perf-output/bisection-poc/gate-artifacts/Isaac-Cartpole-Direct/physx \
  --first_bad_ref ebfc82772 \
  --gpu_model L40S \
  --baselines_dir perf-output/bisection-poc/local_baselines \
  --ensure_baseline \
  --good_fps 1000 \
  --bad_fps 500
```

Convert the gate output into a bisection plan:

```bash
./isaaclab.sh -p tools/perf_regression_gate/bisection_plan_from_gate.py \
  --artifacts_dir perf-output/bisection-poc/gate-artifacts \
  --good_ref HEAD~5 \
  --bad_ref HEAD \
  --gpu_model L40S \
  --baselines_dir perf-output/bisection-poc/local_baselines \
  --output_dir perf-output/bisection-poc/plan \
  --runner_command "{repo_root}/isaaclab.sh -p {repo_root}/tools/perf_regression_gate/bisect_single_commit_runner.py --commit {commit_sha} --task_id {task_id} --backend_key {backend_key} --artifact_dir {artifact_dir} --first_bad_ref ebfc82772 --gpu_model L40S --baselines_dir perf-output/bisection-poc/local_baselines --ensure_baseline --good_fps 1000 --bad_fps 500"
```

Run the bisection harness:

```bash
./isaaclab.sh -p tools/perf_regression_gate/bisection_harness.py run \
  --plan perf-output/bisection-poc/plan/plan.json \
  --output_dir perf-output/bisection-poc/run
```

Generate the diagnosis report:

```bash
./isaaclab.sh -p tools/perf_regression_gate/diagnose_bad_commit.py \
  --run_dir perf-output/bisection-poc/run
```

## Demo Result

The demo found the injected first bad commit:

```json
{
  "status": "completed",
  "reason": "first_bad_found",
  "suspected_first_bad_commit": "ebfc827720ef54a0fd844a01119a0ad1f94546b0",
  "last_good_commit": "237cc6b87870521ce9eb9fba351772bd9a623079"
}
```

The generated diagnosis report shows:

- task/backend
- first bad commit and parent commit
- last tested good commit
- measured last-good FPS
- measured first-bad FPS
- delta vs baseline
- changed files
- changed subsystem guess
- recommended next steps

## Real L40S Demo Result

After validating the synthetic control flow, the POC was exercised with real
IsaacLab benchmark runs on the local L40S host.

The demo branch is `neilm/bisection-real-demo-range`. It contains a short,
deliberate five-commit range:

```text
e1b8f7961 Add bisection real demo notes
0cfef92f2 Describe real bisection demo flow
af9729fe5 Slow Cartpole pre-physics step for demo
291705eae Record demo first bad commit
933ffc9e8 Clarify demo branch tip remains bad
```

The known first-bad commit is:

```text
af9729fe56dbe256aa2a255411fe82814cb9bb50
```

That commit changes only:

```text
source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_direct_env.py
```

and intentionally adds a small per-step delay to
`CartpoleEnv._pre_physics_step`.

### Real Gate Artifact

The full-size local-source runner was run without smoke-test overrides for:

- task: `Isaac-Cartpole-Direct`
- backend: `physx`
- environment count: `4096`
- frame count: `300`
- GPU: `L40S`

A local baseline was seeded from a real known-good run. The bad branch tip then
produced a real gate artifact:

```json
{
  "oracle_verdict": "BLOCK",
  "bisect_verdict": "BAD",
  "baseline_fps": 281105.46120035043,
  "measured_fps": 210768.52333579955,
  "regression_pct": -25.02154798565799,
  "threshold_source": "rolling_window"
}
```

This proves the phase-1 gate artifact can be used as the phase-2 bisection
starting point with real benchmark output.

### Real Bisection Run

The generated plan is:

```text
perf-output/real-bisection-demo/plan/plan.json
```

The completed bisection summary is:

```json
{
  "status": "completed",
  "reason": "first_bad_found",
  "suspected_first_bad_commit": "af9729fe56dbe256aa2a255411fe82814cb9bb50",
  "last_good_commit": "0cfef92f22f6b64702f9984d263fbcf2fcb2444c",
  "tested_commits": [
    "af9729fe56dbe256aa2a255411fe82814cb9bb50",
    "e1b8f7961eada440839071b362f71c3f92a80d9f",
    "0cfef92f22f6b64702f9984d263fbcf2fcb2444c"
  ]
}
```

Candidate evaluations:

- `e1b8f7961ead`: `PASS`, `GOOD`, measured `292955.9` FPS.
- `0cfef92f22f6`: `PASS`, `GOOD`, measured `297987.6` FPS.
- `af9729fe56db`: `BLOCK`, `BAD`, measured `209890.3` FPS.

The diagnosis report is:

```text
perf-output/real-bisection-demo/bisection-run/diagnosis.md
```

It correctly reports:

- first bad commit: `af9729fe56db` - `Slow Cartpole pre-physics step for demo`
- parent/last-good commit: `0cfef92f22f6`
- last-good measured FPS: `297987.6`
- first-bad measured FPS: `209890.3`
- delta vs last-good: `-29.56%`
- delta vs baseline: `-25.33%`
- changed file:
  `source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_direct_env.py`
- changed subsystem: `task/environment config`

### Structured Plan Validation

The original real demo used a raw `runner_command` string in `plan.json`. The
hardened POC now supports a structured v2 plan:

```json
{
  "schema_version": 2,
  "runner": {
    "mode": "local-source",
    "source_dir": "{output_dir}/candidate-source",
    "jit_cache": "{output_dir}/jit-cache",
    "kit_cache": "{output_dir}/kit-cache",
    "ld_preload": "/lib/aarch64-linux-gnu/libgomp.so.1"
  },
  "timeout": {
    "candidate_timeout_s": 900
  },
  "retry": {
    "max_attempts": 2,
    "retry_delay_s": 1
  }
}
```

The structured plan was validated with:

- a synthetic bisection run, confirming the new contract preserves the fast demo
  path
- a real `local-source` run, confirming the harness still finds
  `af9729fe56dbe256aa2a255411fe82814cb9bb50`
- a timeout/retry smoke test, confirming timed-out candidate commands are retried
  and recorded in `results/*.json`

The structured real-run artifacts are under:

```text
perf-output/structured-real-demo/
```

The timeout/retry smoke artifacts are under:

```text
perf-output/timeout-retry-smoke/
```

## What The POC Proves

This POC proves the shape of the phase-2 harness:

- It can consume phase-1 gate artifacts.
- It can create a bisection plan from a regressed cell.
- It can run a midpoint search.
- It can use the existing oracle as the GOOD/BAD/SKIP classifier.
- It can write inspectable state and result artifacts.
- It can generate an initial diagnosis report.
- It can run a real source-isolated bisection on the local L40S host through
  `local-source`.
- It can use a structured v2 `plan.json` with runner, timeout, and retry policy.
- It records per-attempt metadata for timed-out or retried candidates.

## Runner Modes

The single-commit runner now has three modes:

- `synthetic`: keeps the original GPU-free demo path. It uses
  `dev/stub_benchmark.py` to fake the FPS signal while still producing normal
  perf-gate artifacts.
- `docker-source`: checks out the candidate commit into an isolated clone,
  source-mounts that clone into a fixed Docker image, runs one real IsaacLab
  task/backend, and then builds `perf_regression_gate_result.json` with the
  current gate tooling.
- `local-source`: checks out the candidate commit into an isolated clone,
  symlinks the host `env_isaaclab` into that clone, runs the candidate clone's
  own `isaaclab.sh`, and then builds the same result artifact. This is useful on
  local GPU hosts where Docker is unavailable.

The synthetic mode remains useful for quick control-flow demos. The
`docker-source` mode is the intended CI path, while `local-source` is the tested
host path for the current L40S development instance.

Example single-commit invocation:

```bash
./isaaclab.sh -p tools/perf_regression_gate/bisect_single_commit_runner.py \
  --mode docker-source \
  --commit <candidate_sha> \
  --task_id Isaac-Cartpole-Direct \
  --backend_key physx \
  --artifact_dir perf-output/bisection-real/artifacts/<candidate_sha>/Isaac-Cartpole-Direct/physx \
  --image <fixed_isaaclab_ci_image> \
  --gpu_model "RTX PRO 6000"
```

The bisection plan can use the same mode through `runner_command`:

```bash
--runner_command "{repo_root}/isaaclab.sh -p {repo_root}/tools/perf_regression_gate/bisect_single_commit_runner.py --mode docker-source --commit {commit_sha} --task_id {task_id} --backend_key {backend_key} --artifact_dir {artifact_dir} --image <fixed_isaaclab_ci_image> --gpu_model 'RTX PRO 6000'"
```

Example local L40S smoke invocation:

```bash
./isaaclab.sh -p tools/perf_regression_gate/bisect_single_commit_runner.py \
  --mode local-source \
  --commit HEAD \
  --task_id Isaac-Cartpole-Direct \
  --backend_key physx \
  --artifact_dir perf-output/local-source-smoke/artifacts/head/Isaac-Cartpole-Direct/physx \
  --gpu_model L40S \
  --ld_preload /lib/aarch64-linux-gnu/libgomp.so.1 \
  --override_num_envs 16 \
  --override_num_frames 20
```

The override flags are for smoke tests only. Normal bisection runs should omit
them so every commit uses the task shape from `tasks.json`.

The real runner intentionally keeps orchestration outside the historical
checkout:

1. Create an isolated clone or worktree for the candidate commit.
2. Checkout the candidate SHA.
3. Mount that source tree into a fixed dependency image, or symlink the fixed
   host environment for `local-source`.
4. Run one task/backend only.
5. Emit the normal perf-gate artifacts.
6. Keep GPU, dependency environment, task config, seed, and runtime contract
   stable across all tested commits.

This reuses the same idea as baseline seeding: current orchestration code runs
outside, while historical source is materialized separately and mounted or linked
into the benchmark runtime.

## Remaining Real-Run Work

The current real modes provide source-isolated execution paths. The local path is
validated, and a manual Docker/CI validation workflow now exists:

```text
.github/workflows/perf-gate-bisect.yaml
```

The remaining production work is:

- run the Docker/CI workflow on a pushed branch with GPU fleet access
- confirm `docker-source` traverses the known demo range and finds
  `af9729fe56dbe256aa2a255411fe82814cb9bb50`
- tune timeout and retry policy with real CI failure modes
- a decision on whether local development should fix Docker socket permissions or
  keep Docker as CI-only and use `local-source` for host demos

## Later Diagnosis Extensions

After the basic bad-commit report works on real runs, add optional deeper
analysis:

- compare parent vs first-bad frame-time distributions
- compare startup/runtime split
- inspect task-specific config paths
- capture Nsight Systems traces for parent and first bad
- compare CUDA kernel time, memory copies, synchronization stalls, and CPU/GPU
  timelines
- attach trace file paths to `diagnosis.json`

The first version should stay focused on:

> find the bad commit, explain what changed, and recommend where to profile next.
