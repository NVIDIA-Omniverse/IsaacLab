# Range Bisection Invocation

## Request

Required fields are `schema_version`, `operation`, `work_dir`, and
`tooling_ref`. Supply either `good_ref`, `bad_ref`, `task_id`, and
`backend_key`, or a portable resolved `plan`. Set `repo_root` to the target
IsaacLab clone.

```json
{
  "schema_version": 1,
  "operation": "bisect_range",
  "repo_root": "/path/to/IsaacLab",
  "work_dir": "/tmp/isaaclab-bisect",
  "good_ref": "<GOOD_SHA>",
  "bad_ref": "<BAD_SHA>",
  "tooling_ref": "<TOOLING_SHA>",
  "task_id": "Isaac-Velocity-Flat-G1-v0",
  "backend_key": "newton",
  "runner": {
    "mode": "docker-reconstruct",
    "image": "isaaclab-bisection-agent:dev",
    "gpu_model": "NVIDIA L40S",
    "trust_target_code": true,
    "extra_args": [
      "--install_scope",
      "newton,isaacsim"
    ]
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
    "reference_runs": 3,
    "max_reference_runs": 7,
    "candidate_runs": 1,
    "max_candidate_runs": 3,
    "warmup_runs": 1,
    "min_regression_pct": 5.0,
    "gray_zone_pct": 1.0,
    "reference_noise_multiplier": 2.0,
    "max_reference_spread_pct": 10.0,
    "max_tests": 50
  }
}
```

## Response

`result` embeds the canonical `summary.json`.

- `completed`: binary search identified a suspected first-bad commit.
- `inconclusive`: endpoint qualification, tooling compatibility, candidate
  classification, or host execution did not support a culprit.
- `error`: the adapter rejected the request or failed before a canonical
  summary could be interpreted.

The process code is preserved in `exit_code`; automation should make policy
decisions from both `status` and `result.reason`.

## Portable Relaunch

The artifact map includes `plan_resolved` and `relaunch` when available. Copy
the full work directory or at least those files to a stable equivalent host.
The resolved plan is the portable contract; host caches are acceleration only
and are not part of correctness.

Do not change task size, metric, tooling snapshot, or measurement policy during
a relaunch. A changed contract is a new investigation.
