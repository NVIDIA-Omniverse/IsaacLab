<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Compatibility policy

The agent separates three compatibility questions:

1. Can the target commit be checked out?
2. Can its pinned Python, Isaac Sim, and backend stack be reconstructed?
3. Does the pinned benchmark tooling support the target commit's public APIs?

A commit is benchmarkable only when all three checks pass. Setup or tooling
incompatibilities are recorded as skips; they are never converted into `GOOD` or
`BAD` performance verdicts.

## Maintained benchmark window

The initial `0.1` release supports IsaacLab commits that provide the APIs required
by the selected tooling SHA. The capability probe checks these APIs before running
an expensive benchmark and emits `perf_smoke_tooling_incompatible` when the
contract is not satisfied.

Every authoritative run pins one tooling commit and one content hash. The agent
does not silently switch to a candidate-native benchmark implementation because
measurements from different drivers would not be comparable.

## Validated configurations

The current end-to-end validation covered:

- 22 distinct public IsaacLab commits across three non-overlapping ranges.
- `docker-reconstruct` on Linux ARM64.
- NVIDIA L40S with the PhysX Cartpole workload.
- Isaac Sim 6.x / Python 3.12 runtime reconstruction.
- One process warmup and one measured run per commit.
- A cold install from a fresh Ubuntu 24.04 x86_64 host with an NVIDIA RTX PRO
  5000 Blackwell. The six-run Cartpole sample completed in 11 minutes 15 seconds
  with a 302,559.5 FPS median and 5.47% spread.

All 22 ARM64 range commits and the fresh-host Blackwell smoke commit completed
checkout, reconstruction, warmup, benchmark, artifact generation, and
environment cleanup. This is evidence for those configurations, not a claim of
universal compatibility.

Isaac Sim 5.1 / Python 3.11 reconstruction was also exercised against older
commits. Their environment could be reconstructed after applying the legacy
installer adaptations, but the current benchmark tooling contract did not match
their older IsaacLab APIs.

## Release gates

Before expanding the maintained window, add a range probe that covers the new
runtime era and record:

- CPU architecture and GPU model.
- Isaac Sim, Python, physics backend, renderer, Warp, and Newton versions.
- Tooling SHA and bundle hash.
- Per-commit success or structured skip category.
- The generated `probe_range.json` and `report.md`.

The project treats untested GPUs, operating systems, and runtime eras as
best-effort until they are represented in this document and in scheduled GPU
validation.
