# Threshold Check Modes

## CI Gate

`mode: ci_gate` loads a canonical benchmark result and calls the same oracle
used by the CI performance smoke test.

```json
{
  "schema_version": 1,
  "operation": "threshold_check",
  "mode": "ci_gate",
  "bench_result_path": "/tmp/run/perf_smoke_test_result.json",
  "baseline": {
    "median_fps": 60000.0,
    "mad_fps": 500.0,
    "sample_count": 10,
    "source": "perf-baselines"
  },
  "fps_mean_thresholds": [
    {
      "threshold_name": "hard-floor",
      "threshold": 50000.0,
      "threshold_verdict": "BLOCK"
    }
  ],
  "min_block_regression_pct": 5.0,
  "noise_floor_pct": 1.0
}
```

The response `result.verdict` is `PASS`, `WARN`, `BLOCK`, or `HARD_FAILURE`.
`result.bisect_verdict` is the oracle's `GOOD`, `BAD`, or `SKIP` projection.

## Paired Reference Signal

`comparison: check_reference` verifies that measured good and bad endpoints
contain a regression larger than the configured threshold and observed noise.

```json
{
  "schema_version": 1,
  "operation": "threshold_check",
  "mode": "paired_reference",
  "comparison": "check_reference",
  "metric": {
    "name": "raw_fps_mean",
    "result_path": "raw_fps_mean",
    "regression_direction": "decrease",
    "unit": "fps"
  },
  "measurement_policy": {
    "min_regression_pct": 5.0,
    "reference_noise_multiplier": 2.0,
    "max_reference_spread_pct": 10.0
  },
  "good_summary": {
    "label": "good_ref",
    "metric_name": "raw_fps_mean",
    "unit": "fps",
    "values": [60000.0, 60500.0, 59500.0],
    "median_value": 60000.0,
    "mean_value": 60000.0,
    "min_value": 59500.0,
    "max_value": 60500.0,
    "sample_count": 3,
    "spread_pct": 1.24
  },
  "bad_summary": {
    "label": "bad_ref",
    "metric_name": "raw_fps_mean",
    "unit": "fps",
    "values": [54000.0, 54500.0, 53500.0],
    "median_value": 54000.0,
    "mean_value": 54000.0,
    "min_value": 53500.0,
    "max_value": 54500.0,
    "sample_count": 3,
    "spread_pct": 1.37
  }
}
```

## Candidate Comparison

Use `comparison: compare_candidate` with `good_summary`,
`candidate_summary`, and the `reference_noise_pct` from the endpoint check.
The result is `GOOD`, `BAD`, or `UNCLEAR`.

Summary objects are immutable statistics. The adapter intentionally does not
recompute them from `values`; callers should use summaries emitted by the
benchmark or bisection workflow.
