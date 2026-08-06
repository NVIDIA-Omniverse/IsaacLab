# Performance Threshold Check Evaluations

## Scenario 1: Classify An Existing CI Result

Query: "Check this `perf_smoke_test_result.json` against the rolling baseline and hard floor."

Expected behavior:

- Selects `isaaclab-perf-threshold-check` with `mode: ci_gate`.
- Uses the canonical result, supplied baseline statistics, and configured FPS
  thresholds without running Isaac Sim.
- Reports `PASS`, `WARN`, `BLOCK`, or `HARD_FAILURE` exactly as returned and
  preserves the request with the response.

Known failure modes:

- Reruns the benchmark when all required data already exists.
- Recomputes or invents baseline statistics.
- Reinterprets `HARD_FAILURE` as a measured performance regression.

## Scenario 2: Qualify Good And Bad Endpoints

Query: "Do these repeated good and bad FPS measurements separate enough to start bisection?"

Expected behavior:

- Uses `mode: paired_reference` with `comparison: check_reference`.
- Applies the configured minimum regression, observed spread, and noise
  multiplier to the immutable endpoint summaries.
- Proceeds only when the returned reference check says the regression is
  reproduced.

Known failure modes:

- Compares only one sample from each endpoint.
- Ignores excessive endpoint spread.
- Starts binary search after a failed or inconclusive reference check.

## Scenario 3: Classify A Resource Candidate

Query: "Is this candidate's host RAM increase GOOD or BAD relative to the paired references?"

Expected behavior:

- Uses `comparison: compare_candidate` with
  `runtime_resources.system_ram_peak_mb` and regression direction `increase`.
- Uses the supplied good, bad, and candidate summaries without recomputing
  their statistics.
- Preserves `UNCLEAR` when the candidate falls inside the gray/noise zone.

Known failure modes:

- Uses FPS decrease semantics for a resource increase.
- Reads a non-canonical raw resource field.
- Converts `UNCLEAR` into BAD to keep a bisection moving.
