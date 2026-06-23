## Bisection Real Demo Range

This branch is a deliberately small commit range for testing the real bisection
runner.

The intended target is `Isaac-Cartpole-Direct` with the `physx` backend. One
commit in the range intentionally adds extra per-step work to that task so the
local-source runner can find the first bad commit with real benchmark artifacts.
