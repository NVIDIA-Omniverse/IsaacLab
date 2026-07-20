<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# IsaacLab Perf Bisection Agent

Given a known-good and known-bad IsaacLab commit, this agent finds the first commit
that regressed a performance metric. For each tested commit it reconstructs the
commit's own pinned runtime stack, runs one task/backend benchmark, and compares the
result against locally measured good/bad references. Its scope is **IsaacLab commits**;
it reports which pinned components (Isaac Sim, Kit/renderer, PhysX, Newton, Warp) moved
across the range as diagnostic context, but does not bisect inside those components.

## Runner modes

| Mode | Isolation | When to use |
|---|---|---|
| `synthetic` | none (stub) | Rehearse the search over a real range at zero GPU cost. |
| `local-source` | host env | Quick check with your existing IsaacLab env. |
| `docker-source` | existing Docker image | Quick source-mounted check without reconstructing dependencies. |
| `local-reconstruct` | per-commit venv | Faithful, host-independent; rebuilds each commit's pinned stack. |
| `docker-reconstruct` | container + per-commit venv | Strongest isolation; runs the reconstruct flow in a hermetic image. |

`docker-reconstruct` is recommended when you want the host to stay clean and the run to
be reproducible on another machine (e.g. a dedicated CI-matched GPU host).

## Benchmark one commit

The single-commit workflow exposes the reconstruction and benchmark capabilities
without running binary search:

```bash
./isaaclab.sh -p tools/perf_smoke_test/bisection_harness.py benchmark-commit \
    --commit <SHA> \
    --tooling_ref <TOOLING_SHA> \
    --work_dir /tmp/benchmark-commit \
    --runner_mode local-reconstruct \
    --task_id Isaac-Velocity-Flat-G1-v0 \
    --backend_key newton \
    --num_envs 512 \
    --num_frames 300 \
    --warmup_frames 100
```

Authoritative runs require a full committed tooling SHA. The harness resolves that
perf-smoke snapshot before measurement
and records it in `plan.resolved.json` and `tooling_manifest.json`. Every candidate
uses that snapshot's `perf_runtime.py`, result builder, task configuration, metric,
and warmup semantics. Candidate-native benchmark scripts are never selected.

Commits outside the maintained compatibility window may fail because they do not
provide the IsaacLab APIs expected by the pinned tooling. Such failures are reported
as a potential tooling compatibility limitation; the runner does not fall back to a
historical driver or compare results produced by different workflows.
For uncommitted tooling development only, pass `--tooling_ref WORKTREE`; the
result is explicitly non-authoritative and cannot serve as a SHA-only cross-host handoff.

## Docker path: build, run, inspect

### 1. Build the base image

The base image bakes **no** Isaac Sim or pinned stack — that is reconstructed
per-commit at run time. It only provides the OS/CUDA userspace, `git`, and `uv`.

```bash
cd tools/perf_smoke_test/bisection/container
docker build -t isaaclab-bisect:base .
```

### 2. Run the bisection

```bash
./isaaclab.sh -p tools/perf_smoke_test/bisection_harness.py bisect-range \
    --work_dir /tmp/bisect-run \
    --runner_mode docker-reconstruct \
    --image isaaclab-bisect:base \
    --good_ref <GOOD_SHA> \
    --bad_ref <BAD_SHA> \
    --tooling_ref <TOOLING_SHA> \
    --task_id Isaac-Cartpole-Direct \
    --backend_key newton \
    --num_envs 4096 \
    --gpu_model "NVIDIA L40S" \
    --warmup_runs 1
```

`run-local` remains a compatibility alias. Both names run on the current host:
local workstations and Horde/GTL/bare-metal machines use the same implementation.
Hardware matching is advisory and recorded in `hardware_context.json`; the agent
does not provision or automatically escalate to another machine.

For a clean second host, copy `plan.resolved.json`, fetch the
`required_tooling_sha` recorded in `relaunch.json`, and run its `argv` with a new
work directory. The harness archives tooling from that Git object and verifies
the bundle hash before GPU work. It never transfers or substitutes a tooling
bundle:

```bash
git fetch <remote> <TOOLING_SHA>
./isaaclab.sh -p tools/perf_smoke_test/bisection_harness.py bisect-range \
    --plan <copied-plan.resolved.json> --work_dir <new-work-dir>
```

Preflight runs first and writes `preflight.json`: it records the host GPU
(name/driver/VRAM), whether it matches `--gpu_model`, and — for Docker modes — whether
the daemon is reachable and the base image is present. Mismatches/blockers are surfaced
as warnings in `audit_log.jsonl` before any commit is measured.

### 3. Inspect the results

All artifacts land under `--work_dir`:

| File | What it tells you |
|---|---|
| `report.md` | Human-readable verdict, terminal blocker, and component stack diff. |
| `summary.json` | Machine-readable summary (first-bad commit, metric, `stack_diff`). |
| `preflight.json` | Host readiness facts and warnings. |
| `tooling_manifest.json` | Pinned tooling SHA/content hash and measurement contract. |
| `reference_measurements.json` | Good/bad reference medians + reproduction check. |
| `blockers.json` | Structured, categorized blockers with suggested next steps. |
| `artifact_index.json` | Role-based index of every artifact and attempt. |
| `audit_log.jsonl` | Ordered event log (measurements, warmups, recovery, warnings). |
| `measurements/<label>/<sha>/…` | Per-attempt logs and `perf_smoke_test_result.json`. |
| `results/<sha>.json` | Canonical evaluation for each binary-search candidate. |
| `warmup_state.json` | Successful one-per-commit warmups and their stack/tooling identity. |

## What is mounted and cached (docker-reconstruct)

The `docker run` invocation mounts:

- `/harness` (read-only) — the IsaacLab repo providing the harness tooling + git history.
- `/tooling` (read-only) — the run-scoped pinned perf-smoke measurement snapshot.
- `/candidate` — the per-commit source clone the runner checks out and installs editable.
- `/artifacts` — this candidate's artifacts (logs and result).
- `/env-cache` — the **run-scoped, shared** `uv` env cache. Heavy wheels (Isaac Sim,
  Newton, …) are downloaded once and hardlinked/copied across commits, so only the
  first build of a given pinned stack pays the full download cost.
- `/cache/jit-root` and `/cache/kit-root` — run-scoped cache roots. The inner
  runner selects a `stack_hash` subdirectory shared by warmup and measured attempts.

The GPU is provided at run time via `--gpus all`; the host driver is used.

### Cache-sharing note (affects `--warmup_runs`)

Steady-state runs perform one full process warmup per commit by default. The warmup
is recorded and excluded from statistics. Both reconstruct modes use
`{work_dir}/jit-cache/<stack_hash>` and `{work_dir}/kit-cache/<stack_hash>`, so
warmup and measured attempts share compiled kernels and Kit caches. A run-scoped
ledger prevents a commit from being warmed again under another label. Use
`--warmup_runs 0` only when deliberately investigating cold-start behavior.

## Skips and blockers

When a commit cannot be measured, the agent records an honest, categorized skip instead
of guessing a verdict. Host/operator blockers (`host_resource`, `docker_unavailable`,
`gpu_unavailable`, `base_image_missing`) are **not retried** — they point at the machine,
not the commit. Commit/environment skips (`dependency_unavailable`, `install_failed`,
`runtime_incompatible`, `source_checkout_failed`) are handled per their recoverability.
`perf_smoke_tooling_incompatible` is not a normal hole: it stops the active range
without reporting a suspected first-bad commit. It means the candidate lacks APIs
required by the pinned ruler. Tooling SHA/content mismatches are harness-integrity
blockers and also stop before measurement.
See `blockers.json` for the category, reason, and suggested next steps.

FPS is only the default metric. CPU and RAM can be selected directly:

```bash
--metric_path runtime_resources.cpu_util_pct --regression_direction increase --metric_unit %
--metric_path runtime_resources.system_ram_peak_mb --regression_direction increase --metric_unit MB
```
