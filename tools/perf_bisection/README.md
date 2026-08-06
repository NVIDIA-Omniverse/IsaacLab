<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# IsaacLab Bisection Agent

`isaaclab-bisection-agent` finds the first IsaacLab commit that regressed a
performance metric. It checks out each candidate, reconstructs that commit's
pinned runtime stack, runs one fixed benchmark contract, and preserves the
evidence needed to defend the verdict.

The project is currently an alpha developer preview. Deterministic code owns
measurement, thresholding, and binary-search decisions. Optional LLM policies
may diagnose setup failures and propose bounded retries, but cannot change a
`GOOD` or `BAD` verdict. Post-bisection trace analysis remains an external
integration point.

## Install from Isaac Lab

The agent is maintained inside the Isaac Lab repository. Python 3.11 or newer
is required. Use a virtual environment on distributions such as Ubuntu 24.04
that protect the system Python environment. Minimal Ubuntu installations may
need the distribution's `python3-venv` package first:

```bash
python3 -m venv .venv-bisection
.venv-bisection/bin/python -m pip install ./tools/perf_bisection
.venv-bisection/bin/isaaclab-bisect --help
```

For development:

```bash
python3 -m venv .venv-bisection
.venv-bisection/bin/python -m pip install --editable "./tools/perf_bisection[test]"
```

Real benchmark runs require Linux, Git, an NVIDIA GPU/driver compatible with the
target commits, and either Docker with NVIDIA Container Toolkit support or
enough host disk for reconstructed Python environments. Confirm
`docker run --rm --gpus all <cuda-image> nvidia-smi` works before selecting a
Docker runner mode. Follow the official
[Docker Engine](https://docs.docker.com/engine/install/) and
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
installation guides when those prerequisites are absent.

## Build the reconstruction image

The image contains CUDA userspace, build tools, Git, `uv`, and the agent. It
deliberately does not contain Isaac Sim:

```bash
docker build \
    --file tools/perf_bisection/docker/Dockerfile \
    --tag isaaclab-bisection-agent:dev \
    tools/perf_bisection
```

## Benchmark one commit

```bash
isaaclab-bisect benchmark-commit \
    --repo_root /path/to/IsaacLab \
    --commit <COMMIT_SHA> \
    --tooling_ref <TOOLING_SHA> \
    --work_dir /tmp/isaaclab-benchmark \
    --runner_mode docker-reconstruct \
    --trust_target_code \
    --image isaaclab-bisection-agent:dev \
    --task_id Isaac-Cartpole-Direct \
    --backend_key physx \
    --num_envs 4096 \
    --gpu_model "NVIDIA L40S"
```

`--tooling_ref` must be a full committed Isaac Lab SHA for authoritative runs.
The selected commit must contain `tools/perf_smoke_test`; the current checked-out
commit can be selected with `git rev-parse HEAD`. The SHA pins the benchmark
driver, parser, metric semantics, task configuration, and content hash.
`WORKTREE` is available only for explicitly non-authoritative development.

## Probe compatibility across a range

Use `probe-range` to validate checkout, reconstruction, and benchmark execution
without claiming a first-bad commit:

```bash
isaaclab-bisect probe-range \
    --repo_root /path/to/IsaacLab \
    --good_ref <OLDER_SHA> \
    --bad_ref <NEWER_SHA> \
    --tooling_ref <TOOLING_SHA> \
    --work_dir /tmp/isaaclab-probe \
    --runner_mode docker-reconstruct \
    --trust_target_code \
    --image isaaclab-bisection-agent:dev \
    --task_id Isaac-Cartpole-Direct \
    --backend_key physx \
    --num_envs 4096 \
    --cleanup_probe_envs
```

Both endpoints are always measured. `--max_tests` bounds selected interior
commits; set it high enough to cover the complete ancestry path. Each commit is
independent, so setup or benchmark failures are recorded and the sweep
continues. This mode never emits `GOOD`, `BAD`, or a first-bad verdict.

## Bisect a reproduced regression

```bash
isaaclab-bisect bisect-range \
    --repo_root /path/to/IsaacLab \
    --good_ref <GOOD_SHA> \
    --bad_ref <BAD_SHA> \
    --tooling_ref <TOOLING_SHA> \
    --work_dir /tmp/isaaclab-bisect \
    --runner_mode docker-reconstruct \
    --trust_target_code \
    --image isaaclab-bisection-agent:dev \
    --task_id Isaac-Cartpole-Direct \
    --backend_key physx \
    --num_envs 4096 \
    --reference_runs 3 \
    --candidate_runs 1 \
    --warmup_runs 1
```

The run first qualifies the good and bad endpoints. Binary search starts only
when the references are comparable, stable enough, and exceed the effective
regression threshold. A non-reproducing or noisy range is reported as
inconclusive.

## Runner modes

- `synthetic` rehearses search and artifact logic without a GPU.
- `local-source` uses the host's existing IsaacLab environment.
- `docker-source` source-mounts a candidate into an existing IsaacLab image.
- `local-reconstruct` builds each commit's pinned stack on the host.
- `docker-reconstruct` performs that reconstruction in an isolated container
  and is recommended for reproducible investigations.

## Evidence

Important outputs include:

- `plan.resolved.json`: immutable task, runner, metric, sampling, and tooling
  contract.
- `tooling_manifest.json`: tooling SHA, content hash, file count, and contract
  identity.
- `preflight.json` and `hardware_context.json`: host and GPU evidence.
- `reference_measurements.json`: endpoint samples and qualification result.
- `summary.json` and `report.md`: final verdict, values, thresholds, confidence,
  narrowed interval, stack movement, and blockers.
- `probe_range.json`: non-authoritative compatibility sweep results.
- `measurements/` and `results/`: attempt logs and per-candidate evidence.
- `relaunch.json`: an argv handoff for an equivalent host.
- `security_scan.json`: value-free credential finding locations; a `blocked`
  status prohibits sharing until reviewed and remediated.

Setup and tooling incompatibilities are structured skips, not performance
verdicts. See [the compatibility policy](docs/compatibility.md).

## Agent Skills

The repository ships operator and automation playbooks under the native
[`skills/`](../../skills/README.md) catalog. The three atomic operations share
one JSON adapter:

```bash
isaaclab-bisect-skill --input request.json --output response.json
```

The adapter preserves the canonical artifacts and returns a small response
envelope for Fanes Agent or other automation.

Optional official IsaacLab/Isaac Sim Skills can provide host onboarding,
backend selection, setup troubleshooting, and post-bisection profiling. Their
reviewed sources are commit-pinned and they never run in the candidate or verdict
paths. See [upstream Skill integration](docs/upstream-skills.md).

## Development and release

- [Development and validation](docs/development.md)
- [Compatibility policy](docs/compatibility.md)
- [Optional LLM policies](docs/llm-policies.md)
- [Optional upstream Skills](docs/upstream-skills.md)
- [NVIDIA security readiness](docs/security-readiness.md)
- [Security threat model](docs/security-threat-model.md)
- [Security test plan](docs/security-test-plan.md)
- [Draft agent and Skill cards](docs/agent-card.md), [Skill cards](docs/skill-cards.md)
- [Release process](docs/releasing.md)

The core has no required third-party Python dependency. Tests, linting, and
container builds run in GitHub Actions; authoritative GPU sweeps remain a
separate hardware validation gate.
