# Bisection Agent Bootstrap

This note explains how YuTeh's regression-agent pattern maps to phase 2 of the
perf-gate project and how to start with a preliminary bisection agent.

## What YuTeh's Note Means

YuTeh described an existing OmniPerf flow:

1. A Slack-facing agent discovers benchmark configurations in a database.
2. It writes a `plan.json` and `template.sql`.
3. It launches `regression-agent` in "Phase 2 only" mode.
4. The agent scans already-collected benchmark data.
5. It writes `status.json` and `results/*.json`.
6. The Slack agent summarizes any regressions.

The key detail is the `yts/no_grafana` branch. That branch skips the old Grafana
dashboard-fetching step and goes straight to regression detection. In plain
terms: if benchmark data already exists somewhere else, the agent does not need
Grafana as an input.

## How That Relates To Our Phase 2

Our phase 1 perf gate answers:

> Did this PR regress performance for a task/backend?

The phase 2 bisection agent should answer:

> Which commit introduced that regression?

So YuTeh's agent and our agent are related, but not identical.

YuTeh's regression-agent:

- Works on historical rows that already exist in a benchmark database.
- Scans many configs in parallel.
- Detects whether metric trends changed.
- Produces regression reports.

Our bisection agent:

- Starts after the perf gate has already found a regression.
- Focuses on one regressed task/backend first.
- Tests commits between a known-good SHA and known-bad SHA.
- Reuses the perf-gate oracle to classify each commit as `GOOD`, `BAD`, or `SKIP`.
- Produces a suspected first bad commit plus evidence.

## Design Principle To Copy

The most useful idea to copy is the file contract:

- `plan.json`: what to analyze
- `status.json`: current progress
- `results/*.json`: per-config or per-commit results
- `summary.json`: final answer

That makes the agent easy to run from a CLI, Slack bot, GitHub workflow, or future
automation.

## Design Principle To Avoid

Do not mix incompatible populations.

YuTeh's note warns that pooling multiple `comparison_group_name` values can
produce fake regressions. For us, the equivalent rule is:

- Do not mix task IDs.
- Do not mix backend keys.
- Do not mix GPU models.
- Do not mix launch configs.
- Do not mix runtime contracts.
- Do not change the image/runtime environment halfway through a bisection.

If any of those change, the agent may blame the wrong commit.

## Preliminary Agent

The bootstrap script is:

```bash
tools/perf_regression_gate/bisect_agent.py
```

It currently provides the orchestration shell:

1. Resolve `good_ref` and `bad_ref`.
2. List candidate commits on the ancestry path.
3. Pick midpoint commits.
4. Run a caller-provided command for each candidate.
5. Read normal perf-gate artifacts from that command.
6. Ask the existing oracle for `GOOD`, `BAD`, or `SKIP`.
7. Write `status.json`, `results/*.json`, and `summary.json`.

This lets us refine the expensive part, "how exactly do we run one commit," later
without rewriting the bisection control flow.

The demo single-commit runner is:

```bash
tools/perf_regression_gate/bisect_single_commit_runner.py
```

It is intentionally synthetic for now. It uses `dev/stub_benchmark.py` to write
normal perf-gate artifacts without launching Isaac Sim. The demo knob is
`--first_bad_ref`: that ref and its descendants emit low FPS, while earlier
commits emit healthy FPS.

## Plan Shape

Example `plan.json`:

```json
{
  "task_id": "Isaac-Velocity-Flat-G1-v0",
  "backend_key": "physx",
  "good_ref": "6c76e4a068ca8456618de70d5fbfc5ee3ed2364e",
  "bad_ref": "38197210a",
  "gpu_model": "rtx_pro_6000_blackwell",
  "baselines_dir": "tools/perf_regression_gate/local_baselines",
  "gate_config": "tools/perf_regression_gate/gate_config.json",
  "runner_command": "python3 tools/perf_regression_gate/scripts/run_one_commit_placeholder.py --commit {commit_sha} --task {task_id} --backend {backend_key} --artifact-dir {artifact_dir}"
}
```

Supported runner-command placeholders:

- `{commit_sha}`
- `{task_id}`
- `{backend_key}`
- `{artifact_dir}`
- `{repo_root}`

The runner command must write these files under `{artifact_dir}`:

- `perf_regression_gate_result.json`
- `perf_regression_gate_info.json`
- `benchmark.log`
- `launch_config.json`

That is the same artifact contract phase 1 already uses.

## Synthetic End-To-End Demo

This command demonstrates the full control loop without using GPUs:

```bash
./isaaclab.sh -p tools/perf_regression_gate/bisect_agent.py \
  --good_ref HEAD~5 \
  --bad_ref HEAD \
  --task_id Isaac-Cartpole-Direct \
  --backend_key physx \
  --gpu_model L40S \
  --baselines_dir perf-output/bisect-synthetic-demo/local_baselines \
  --output_dir perf-output/bisect-synthetic-demo \
  --runner_command "{repo_root}/isaaclab.sh -p {repo_root}/tools/perf_regression_gate/bisect_single_commit_runner.py --commit {commit_sha} --task_id {task_id} --backend_key {backend_key} --artifact_dir {artifact_dir} --first_bad_ref ebfc82772 --gpu_model L40S --baselines_dir perf-output/bisect-synthetic-demo/local_baselines --ensure_baseline --good_fps 1000 --bad_fps 500"
```

What this does:

1. Builds the candidate list from `HEAD~5..HEAD`.
2. Pretends `ebfc82772` is the first bad commit.
3. Emits roughly 1000 FPS before `ebfc82772`.
4. Emits roughly 500 FPS at `ebfc82772` and descendants.
5. Seeds a matching local baseline under `perf-output/bisect-synthetic-demo/local_baselines`.
6. Uses the normal perf-gate oracle to classify commits.
7. Writes the suspected first bad commit to `summary.json`.

Example output from this POC:

```json
{
  "reason": "first_bad_found",
  "status": "completed",
  "suspected_first_bad_commit": "ebfc827720ef54a0fd844a01119a0ad1f94546b0",
  "tested_commits": [
    "ebfc827720ef54a0fd844a01119a0ad1f94546b0",
    "3d6b2ecad53464d3fb8c9982569beaccdeaefb2a",
    "237cc6b87870521ce9eb9fba351772bd9a623079"
  ]
}
```

Per-commit result files live under:

```text
perf-output/bisect-synthetic-demo/results/
```

For example, the injected bad commit records:

```json
{
  "bisect_verdict": "BAD",
  "oracle_verdict": "BLOCK",
  "baseline_fps": 1000.0,
  "measured_fps": 500.0,
  "regression_pct": -50.0
}
```

## Dry Run

Use dry run first to verify the commit range:

```bash
python3 tools/perf_regression_gate/bisect_agent.py \
  --good_ref <known-good-sha> \
  --bad_ref <known-bad-sha> \
  --task_id Isaac-Velocity-Flat-G1-v0 \
  --backend_key physx \
  --runner_command "echo placeholder" \
  --output_dir /tmp/perf-bisect-demo \
  --dry_run
```

This writes:

```text
/tmp/perf-bisect-demo/
  plan.resolved.json
  candidates.json
  status.json
```

## Runner Modes

The single-commit runner has three execution modes.

`synthetic` is the default demo mode. It runs `dev/stub_benchmark.py`, fakes the
FPS value, and emits normal perf-gate artifacts. This keeps the bisection loop
fast and GPU-free while validating the harness contract.

`docker-source` is the real single-cell mode. It checks out the candidate commit
into an isolated clone, source-mounts that clone into a fixed IsaacLab CI image,
runs exactly one task/backend, and then lets the current gate tooling build the
normal result artifact. This is the intended CI path.

`local-source` is the host-backed real mode. It checks out the candidate commit
into an isolated clone, symlinks the current host `env_isaaclab` into that clone,
runs the candidate clone's own `isaaclab.sh`, and writes the same artifacts. This
is useful on local GPU instances where Docker is not available to the runner
user.

Example:

```bash
./isaaclab.sh -p tools/perf_regression_gate/bisect_single_commit_runner.py \
  --mode docker-source \
  --commit <candidate_sha> \
  --task_id Isaac-Velocity-Flat-G1-v0 \
  --backend_key physx \
  --artifact_dir /tmp/perf-bisect-demo/artifacts/<candidate_sha>/Isaac-Velocity-Flat-G1-v0/physx \
  --image <fixed_isaaclab_ci_image> \
  --gpu_model "RTX PRO 6000"
```

To plug this into a generated plan:

```bash
--runner_command "{repo_root}/isaaclab.sh -p {repo_root}/tools/perf_regression_gate/bisect_single_commit_runner.py --mode docker-source --commit {commit_sha} --task_id {task_id} --backend_key {backend_key} --artifact_dir {artifact_dir} --image <fixed_isaaclab_ci_image> --gpu_model 'RTX PRO 6000'"
```

Example local L40S smoke run:

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

The override flags are only for smoke tests. Normal bisection runs should omit
them and use the task shape from `tasks.json`.

The runner reuses the same source isolation idea as baseline seeding:

- the bisection harness and oracle run from the current checkout
- historical candidate source is materialized in a separate clone
- the candidate clone is mounted over `/workspace/isaaclab` in the container, or
  linked to the host `env_isaaclab` in `local-source` mode
- JIT and Kit cache directories are shared across candidates to avoid avoidable
  warm-up cost
- artifacts are written outside the candidate clone, so each commit can be
  evaluated independently

## What We Still Need To Build

The preliminary agent now has real single-commit execution paths. The remaining
work is around productionizing them:

1. **CI wrapper**
   - Add a manual GitHub workflow that can launch the bisection plan on the GPU
     fleet and upload the output directory.

2. **Image/cache policy**
   - Decide which fixed CI image the bisection workflow should use and whether it
     should pull from GHCR, NVCR, or a runner-local cache.

3. **Failure policy**
   - Add retry handling for infra-only failures so one flaky Docker/GPU startup
     does not prematurely classify a commit as bad.

4. **Real regression demo**
   - Run against a deliberately regressed IsaacLab branch to prove the
     real runner path finds the same first bad commit the synthetic demo
     currently finds.

## Why Start This Way

This is small enough to build now but still points in the final direction.

It gives us:

- A concrete command to run.
- A concrete plan format.
- A concrete status/result format.
- A place to plug in real commit execution.
- A simple story for future Slack/GitHub automation.

It avoids prematurely solving:

- Final Docker image strategy.
- Full GitHub workflow integration.
- Parallel bisection.
- Database-backed historical analysis.
- Multi-task bisection.

Those can come after the single-cell, single-range loop works.
