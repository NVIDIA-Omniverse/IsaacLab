# Benchmark One Commit Evaluations

## Scenario 1: Authoritative Historical Measurement

Query: "Benchmark commit `<SHA>` on the L40S using the current perf-smoke ruler."

Expected behavior:

- Selects `isaaclab-perf-benchmark-commit`.
- Creates a schema-valid `benchmark_commit` request with full candidate and
  tooling SHAs, a fresh work directory, and one fixed task/backend/metric
  contract.
- Uses `docker-reconstruct` for the authoritative run and accepts success only
  when `status` is `completed` and `result.succeeded` is true.
- Preserves the response and complete work directory as one audit unit.

Known failure modes:

- Uses `WORKTREE` tooling while describing the result as authoritative.
- Invokes benchmark code from the candidate checkout instead of the pinned
  tooling snapshot.
- Treats process exit code zero as sufficient without checking the result.

## Scenario 2: Local Tooling Development

Query: "Test my uncommitted bisection tooling against one commit before I commit it."

Expected behavior:

- Uses `tooling_ref: WORKTREE` and clearly labels the result
  non-authoritative.
- Keeps the candidate commit, task, metric, and measurement policy explicit.
- Does not present the output as a portable cross-host measurement.

Known failure modes:

- Requires a committed tooling SHA for this explicitly developmental run.
- Omits the non-authoritative limitation.
- Reuses the output as an authoritative reference in a later bisection.

## Scenario 3: Candidate Cannot Be Measured

Query: "The single-commit benchmark failed during reconstruction. Is that commit a regression?"

Expected behavior:

- Reads the response status, `result.succeeded`, categorized skip or blocker,
  and linked attempt artifacts.
- Reports an environment, host, or tooling failure as evidence that no metric
  was obtained.
- Does not classify the commit as a performance regression.

Known failure modes:

- Converts any installation or runtime failure into a BAD verdict.
- Falls back to a different benchmark ruler to obtain a number.
- Discards the install log, tooling manifest, or attempt summary.
