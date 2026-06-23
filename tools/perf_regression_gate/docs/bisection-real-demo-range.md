## Bisection Real Demo Range

This branch is a deliberately small commit range for testing the real bisection
runner.

The intended target is `Isaac-Cartpole-Direct` with the `physx` backend. One
commit in the range intentionally adds extra per-step work to that task so the
local-source runner can find the first bad commit with real benchmark artifacts.

Expected demo flow:

1. Run the target cell at a known-good commit to seed a local baseline.
2. Run the target cell at the branch tip to produce a regressed gate artifact.
3. Feed that artifact into the bisection harness.
4. Confirm the harness identifies the slowdown commit as the first bad commit.

Known first-bad commit for this branch:

- `af9729fe56dbe256aa2a255411fe82814cb9bb50`
