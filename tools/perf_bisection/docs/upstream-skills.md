<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Optional upstream Skills

The bisection agent composes with four official Skills that provide useful
operator guidance around the deterministic workflow:

- `isaaclab-installing-isaac-lab` for initial host onboarding or a current
  checkout used by `local-source`.
- `isaaclab-setup-troubleshooting` for host or current-checkout setup failures.
- `isaaclab-selecting-backends` when the task/backend combination is unclear.
- `profile-isaac-sim` for deeper profiling after bisection identifies a culprit.

These Skills are not Python dependencies and are never executed by the bisection
engine. They cannot classify metrics, discard a search interval, or change a
`GOOD`, `BAD`, or `UNCLEAR` verdict.

The installation Skill must not replace per-commit reconstruction. It follows
the current checkout's installation documentation, while `docker-reconstruct`
must reproduce each historical commit's own pinned stack unattended.

## Pinned sources

`tools/perf_bisection/src/isaaclab_bisection/upstream_skills.lock.json` records reviewed commit SHAs,
Skill paths, handoff points, restrictions, and the exact Skills CLI version plus
npm integrity value. Updating any pin is a normal reviewed code change; upstream
branch or package movement never changes an existing agent release.

Validate and inspect the lock:

```bash
isaaclab-bisect-upstream-skills validate
isaaclab-bisect-upstream-skills list
```

## Installation

The Skills CLI requires Node.js and `npx`. Print version-pinned installation
commands for Cursor:

```bash
isaaclab-bisect-upstream-skills commands --agent cursor
```

To install only one optional Skill:

```bash
isaaclab-bisect-upstream-skills commands \
    --agent cursor \
    --skill isaaclab-selecting-backends
```

Review the pinned CLI version and generated source URLs before running the
commands. `npx --yes` suppresses only npm's package-download prompt; the Skills
CLI's own installation confirmation remains enabled. The bisection package
prints commands but does not install or update external Skills itself.

Private NVIDIA Skill repositories are intentionally not required by the public
Isaac Lab workflow. Internal deployments may install signed equivalents
separately while preserving the same handoff names and restrictions.
