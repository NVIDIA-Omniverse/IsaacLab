# Performance Range Bisection Evaluations

## Scenario 1: Authoritative First-Bad Search

Query: "Bisect this known FPS regression between `<GOOD_SHA>` and `<BAD_SHA>` on the L40S."

Expected behavior:

- Selects `isaaclab-perf-bisect-range`.
- Creates a schema-valid request with full good, bad, and tooling SHAs,
  `docker-reconstruct`, and one fixed task/backend/metric/measurement contract.
- Qualifies both endpoints before binary search.
- Accepts a culprit only when `status` is `completed` and
  `result.suspected_first_bad_commit` is populated.

Known failure modes:

- Begins binary search before measuring both references.
- Changes the task, tooling, metric, or threshold during the search.
- Reports the last measured commit as the culprit without a completed result.

## Scenario 2: Regression Does Not Reproduce

Query: "The nominal bad endpoint is as fast as the good endpoint. Which commit caused the regression?"

Expected behavior:

- Reports the range result as `inconclusive`.
- Preserves endpoint measurements, spread, hardware context, and the returned
  reason.
- States that no first-bad commit was established on this host and contract.

Known failure modes:

- Forces GOOD/BAD endpoint labels from their names.
- Lowers the regression threshold after seeing the measurements.
- Fabricates a culprit from commit history or source diff alone.

## Scenario 3: Pinned Tooling Is Incompatible

Query: "Continue past this historical commit even though it lacks the API required by the pinned perf tooling."

Expected behavior:

- Treats `perf_smoke_tooling_incompatible` as a terminal support-window
  boundary.
- Returns the unsupported or inconclusive result and preserves the blocker
  artifacts.
- Recommends a separately versioned compatibility effort if the historical
  range must be supported.

Known failure modes:

- Skips around the commit as if it were a normal measurement hole.
- Invokes candidate-native or historical benchmark tooling.
- Compares results produced by different rulers.

## Scenario 4: Relaunch On Equivalent Hardware

Query: "Move this investigation to another equivalent GPU host without changing the experiment."

Expected behavior:

- Uses the existing `plan.resolved.json` and `relaunch.json` with a fresh work
  directory.
- Fetches and verifies the recorded tooling SHA and content hash.
- Keeps task size, metric, tooling, and measurement policy unchanged.

Known failure modes:

- Copies an unverified tooling bundle instead of fetching the Git object.
- Treats host caches as part of the correctness contract.
- Modifies the plan to make the new host produce a preferred verdict.
