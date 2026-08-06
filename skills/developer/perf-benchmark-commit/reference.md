# Single-Commit Benchmark Invocation

## Request

Required fields are `schema_version`, `operation`, `work_dir`, `commit`,
`tooling_ref`, `task_id`, and `backend_key`. Set `repo_root` to the target
IsaacLab clone. `operation` is always
`benchmark_commit`.

The `runner`, `task`, `metric`, and `measurement` objects map directly to
supported `benchmark-commit` harness flags. Omitted optional fields use harness
defaults. A full `plan` may supply task and runner defaults, but `commit`,
`work_dir`, and `tooling_ref` remain explicit for auditability.

Example:

```json
{
  "schema_version": 1,
  "operation": "benchmark_commit",
  "repo_root": "/path/to/IsaacLab",
  "work_dir": "/tmp/isaaclab-benchmark",
  "commit": "<COMMIT_SHA>",
  "tooling_ref": "<TOOLING_SHA>",
  "task_id": "Isaac-Velocity-Flat-G1-v0",
  "backend_key": "newton",
  "runner": {
    "mode": "docker-reconstruct",
    "image": "isaaclab-bisection-agent:dev",
    "gpu_model": "NVIDIA L40S",
    "trust_target_code": true
  },
  "task": {
    "num_envs": 512,
    "num_frames": 300,
    "warmup_frames": 100
  },
  "metric": {
    "name": "raw_fps_mean",
    "result_path": "raw_fps_mean",
    "regression_direction": "decrease",
    "unit": "fps"
  },
  "measurement": {
    "runs": 3,
    "max_runs": 7,
    "warmup_runs": 1
  }
}
```

## Response

`status` is `completed` only when `measurement_summary.json` reports
`succeeded: true`. `inconclusive` means no canonical summary was produced.
`error` means the adapter rejected the request or could not invoke the harness.

`exit_code` preserves the harness process code. `result` embeds
`measurement_summary.json`; `artifacts` contains absolute paths to files that
exist when the response is written.

The adapter does not copy raw logs into the response. Follow paths in
`result.attempts` and `artifacts` to inspect evidence.

## Automation Contract

- Invoke through
  `$REPO_ROOT/.venv-bisection/bin/isaaclab-bisect-skill` from the current Isaac
  Lab checkout.
- Give every invocation a fresh `work_dir`, or deliberately resume the same run.
- Parse the output file, not stdout.
- Preserve the entire response and work directory as one audit unit.
- Do not infer success from process code alone; check `status` and
  `result.succeeded`.
