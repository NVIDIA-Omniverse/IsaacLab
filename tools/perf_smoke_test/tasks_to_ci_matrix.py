# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert tasks.json into the GitHub Actions bench matrix JSON.

Prints a JSON array to stdout, one object per (task_id, backend) combination,
containing the fields consumed by the ``bench`` job matrix in perf-smoke-test.yaml.

Each row is optionally *enriched* with golden-correctness fields when a matching
(task_id, backend) exists in ``golden_tasks.json``: the golden rollout runs as an
optional second stage inside the same bench job (same warm runner, no extra image
pull), so its run-shape rides on the same matrix cell and ``job_timeout_minutes``
auto-includes the golden budget. Golden is fully optional -- absent/parse-failing
golden config simply leaves ``golden_present`` false and the perf budget unchanged.

CAVEAT (golden requires a perf pair): because golden piggybacks the perf ``bench``
cells, a golden task runs ONLY for a (task_id, backend) that ALSO has a perf entry
in ``tasks.json``. A golden config with no matching perf entry is never placed in
the matrix and simply does not run -- there is deliberately no standalone golden
job (that would force a guaranteed cold image pull on the ephemeral runner fleet,
the exact cost the piggyback design eliminates). If a golden-only task is ever
needed, add a (cheap) perf entry for it, or extend the matrix to perf-union-golden
with perf-optional cells.

Usage::

    python3 tools/perf_smoke_test/tasks_to_ci_matrix.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from launch_config import hydra_args_for_task  # noqa: E402
from task_config import load_tasks  # noqa: E402

# Golden enrichment is best-effort: a broken/absent golden config must never break
# perf matrix generation (golden is an optional add-on).
try:
    from golden_config import get_golden_task

    _GOLDEN_AVAILABLE = True
except Exception:  # pragma: no cover - golden is optional
    _GOLDEN_AVAILABLE = False

# Buffer minutes added on top of a rollout's hard timeout for retry + overhead,
# mirroring the perf gate's existing "+15" job-timeout headroom.
_PERF_JOB_BUFFER_MIN = 15
_GOLDEN_JOB_BUFFER_MIN = 5


def _golden_for(task):
    """Return the matching GoldenTaskConfig for a perf task, or None (best-effort)."""
    if not _GOLDEN_AVAILABLE:
        return None
    try:
        return get_golden_task(task.task_id, task.backend_key)
    except Exception:
        return None


tasks = load_tasks()
rows = []
for task in tasks:
    row = {
        "task_id": task.task_id,
        "physics_backend": task.physics_backend,
        "render_backend": task.render_backend or "",
        "num_envs": task.num_envs,
        "num_frames": task.num_frames,
        "warmup_frames": task.warmup_frames,
        "seed": task.seed if task.seed is not None else "",
        "hydra_args": " ".join(hydra_args_for_task(task)),
        "bench_timeout_s": task.timeout_minutes * 60,
    }

    # job_timeout_minutes auto-calculates from the perf budget plus the golden
    # budget (0 when this cell has no golden config), so the shared bench job has
    # headroom for perf(run+retry) + optional golden(run+retry).
    golden = _golden_for(task)
    golden_budget_min = 0
    if golden is not None:
        row["golden_present"] = "true"
        row["golden_num_envs"] = golden.num_envs
        row["golden_eval_steps"] = golden.eval_steps
        row["golden_seed"] = golden.seed if golden.seed is not None else ""
        row["golden_hydra_args"] = " ".join(hydra_args_for_task(golden))
        row["golden_checkpoint_id"] = golden.checkpoint_id
        row["golden_checkpoint_relpath"] = golden.checkpoint_relpath
        row["golden_timeout_s"] = golden.timeout_minutes * 60
        golden_budget_min = golden.timeout_minutes + _GOLDEN_JOB_BUFFER_MIN
    else:
        row["golden_present"] = "false"

    perf_budget_min = task.timeout_minutes + _PERF_JOB_BUFFER_MIN
    row["job_timeout_minutes"] = max(30, perf_budget_min + golden_budget_min)
    rows.append(row)

print(json.dumps(rows))
