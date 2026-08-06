# Performance Bisection Reference

## Contents

- Single-commit workflow
- Range workflow
- Automation workflow
- Pinned tooling and support window
- Artifacts
- Hardware guidance
- Validation commands

## Single-Commit Workflow

Use the host directly:

```bash
isaaclab-bisect benchmark-commit \
    --repo_root /path/to/IsaacLab \
    --commit <SHA> \
    --tooling_ref <TOOLING_SHA> \
    --work_dir /tmp/benchmark-commit \
    --runner_mode local-reconstruct \
    --trust_target_code \
    --task_id Isaac-Velocity-Flat-G1-v0 \
    --backend_key newton \
    --num_envs 512 \
    --num_frames 300 \
    --warmup_frames 100 \
    --gpu_model "NVIDIA L40S"
```

Use `docker-reconstruct` plus `--image isaaclab-bisection-agent:dev` for container
isolation. Both modes reconstruct the commit's pinned runtime stack.

## Range Workflow

The same range command runs locally or on a dedicated host:

```bash
isaaclab-bisect bisect-range \
    --repo_root /path/to/IsaacLab \
    --good_ref <GOOD_SHA> \
    --bad_ref <BAD_SHA> \
    --tooling_ref <TOOLING_SHA> \
    --work_dir /tmp/bisect-run \
    --runner_mode docker-reconstruct \
    --trust_target_code \
    --image isaaclab-bisection-agent:dev \
    --task_id Isaac-Velocity-Flat-G1-v0 \
    --backend_key newton \
    --num_envs 512 \
    --num_frames 300 \
    --warmup_frames 100 \
    --gpu_model "NVIDIA L40S"
```

`run-local` remains a compatibility alias for `bisect-range`.

## Automation Workflow

The three atomic Skills share one versioned adapter:

```bash
isaaclab-bisect-skill \
    --input <request.json> \
    --output <response.json>
```

Set `operation` to `benchmark_commit`, `threshold_check`, or `bisect_range`.
Use the input and output schemas linked by the corresponding atomic Skill:

- `isaaclab-perf-benchmark-commit` for one commit.
- `isaaclab-perf-threshold-check` for an existing result or summaries.
- `isaaclab-perf-bisect-range` for endpoint qualification and binary search.

The adapter preserves canonical harness artifacts rather than creating a second
result format. Its response is a small envelope containing status, process code,
the primary canonical result, and artifact paths. Fanes Agent and other
automation should parse the response file and archive the work directory.

## Upstream Skill Handoffs

The reviewed upstream Skill pins are stored in
`tools/perf_bisection/src/isaaclab_bisection/upstream_skills.lock.json`.

- Use `isaaclab-installing-isaac-lab` before measurement only when the operator
  needs a current host installation. It does not reconstruct historical commits.
- Use `isaaclab-setup-troubleshooting` for host/current-checkout failures, not to
  reinterpret candidate skips.
- Use `isaaclab-selecting-backends` before resolving the plan when backend choice
  is unclear.
- Use `profile-isaac-sim` after a culprit is identified and only when its
  release-build profiling workflow applies.

Generate immutable installation commands with
`isaaclab-bisect-upstream-skills commands --agent cursor`.

## Pinned Tooling and Support Window

Authoritative runs name a full committed tooling SHA. The harness resolves that
perf-smoke snapshot before a run. Its
tooling commit/content hash, fixed `perf_runtime.py` driver, RuntimeBundle result
schema, resolved task configuration, selected metric, frame warmup, and process
warmup are persisted in `plan.resolved.json`. The snapshot is materialized under
the work directory and mounted read-only in Docker.

Candidates provide IsaacLab source and their pinned runtime dependencies. They
do not provide the benchmark driver or parser. The supported window begins when
the candidate exposes the IsaacLab benchmark APIs required by the maintained
driver and is intended for normal days-to-weeks regression investigations.
Older incompatible commits stop cleanly as a potential tooling-compatibility
limitation; there is no candidate-native fallback.
The terminal status is `unsupported_tooling_contract` with reason
`perf_smoke_tooling_incompatible`; the search does not step around it as a hole.
`WORKTREE` is available only for non-authoritative local development.

FPS is the default metric, not the only selectable metric. Any numeric field
projected into `perf_smoke_test_result.json` can be selected with `--metric_path`
and the correct `--regression_direction`. GPU memory uses
`gpu_diag.gpu_mem_used_mb`; CPU utilization uses
`resource_diag.cpu_util_pct_mean`; peak host RAM uses
`resource_diag.ram_used_gb_peak`. Resource regressions use direction `increase`.

## Artifacts

Single-commit runs produce:

- `plan.resolved.json`: portable resolved inputs.
- `tooling_manifest.json`: pinned tooling and measurement-contract identity.
- `preflight.json`: GPU, driver, VRAM, disk, Docker, and image checks.
- `hardware_context.json`: advisory target match label.
- `measurement_summary.json`: canonical metric summary and attempts.
- `relaunch.json`: copyable argv for another local or dedicated host.
- `measurements/`: raw per-attempt logs and benchmark artifacts.
- `results/`: per-candidate binary-search evaluations.
- `warmup_state.json`: one-per-commit warmup ledger and cache identity.

Range runs additionally produce `reference_measurements.json`, `summary.json`,
`report.md`, `blockers.json`, and the binary-search results.

## Hardware Guidance

Local full bisection is supported. Use stable dedicated GPU hardware when
the result must match CI or when a local GPU lacks memory. Hardware selection
does not change the command or measurement implementation. Automatic SSH,
provisioning, scheduling, and escalation are outside this workflow.

## Validation Commands

```bash
python -m pytest -q
```

```bash
ruff check .
ruff format --check .
```

Run both checks before committing.
