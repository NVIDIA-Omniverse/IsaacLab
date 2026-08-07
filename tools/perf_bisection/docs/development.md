<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Development

## Local setup

From the Isaac Lab repository root, create an environment with Python 3.11 or
newer, then install the package:

```bash
uv venv --python 3.12 .venv-bisection
source .venv-bisection/bin/activate
uv pip install --python .venv-bisection/bin/python --editable "./tools/perf_bisection[test]"
```

Run the CPU-only validation:

```bash
ruff check tools/perf_bisection
ruff format --check tools/perf_bisection
pytest -q tools/perf_bisection/tests
isaaclab-bisect --help
isaaclab-bisect-upstream-skills validate
uv run --no-project python tools/skills/cli.py check
```

Unit and synthetic tests must not require Isaac Sim, a GPU, or Docker.
Scan any generated run directory with
`isaaclab-bisect-scan-artifacts <OUTPUT_DIR>` before sharing it.
The scanner excludes reconstructed environments, caches, and source clones;
never include those excluded directories in a shared evidence archive.

## Container

Build the reconstruction image from the repository root:

```bash
docker build \
    --file tools/perf_bisection/docker/Dockerfile \
    --tag isaaclab-bisection-agent:dev \
    tools/perf_bisection
```

The image intentionally contains no Isaac Sim installation. Each candidate's
pinned runtime is reconstructed into a run-scoped cache.

## GPU validation

GPU validation is intentionally separate from pull-request CI. Use
`probe-range` first because it records every per-commit compatibility result and
continues after setup failures:

```bash
isaaclab-bisect probe-range \
    --repo_root /path/to/IsaacLab \
    --work_dir /tmp/isaaclab-probe \
    --runner_mode docker-reconstruct \
    --trust_target_code \
    --image isaaclab-bisection-agent:dev \
    --good_ref <OLDER_SHA> \
    --bad_ref <NEWER_SHA> \
    --tooling_ref <TOOLING_SHA> \
    --task_id Isaac-Cartpole-Direct \
    --backend_key physx \
    --num_envs 4096 \
    --cleanup_probe_envs
```

Do not use a probe report as regression evidence. After compatibility is
established, run `bisect-range` with the normal reference and candidate sampling
policy.

## Contribution requirements

- Keep deterministic verdict logic independent from LLM recovery and diagnosis.
- Add regression tests for every newly supported setup failure.
- Verify a bug regression test fails without its fix.
- Preserve structured skip categories instead of collapsing failures into
  benchmark regressions.
- Treat candidate source/logs and all model output as untrusted. Enforce
  security policy in deterministic code, never only in a prompt.
- Do not add a credential, mount, egress destination, shared-resource write, or
  model-controlled action without updating the threat model and security tests.
- Update `docs/compatibility.md` when extending the validated matrix.
