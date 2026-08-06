---
name: isaaclab-perf-bisection
description: Benchmarks supported Isaac Lab commits with pinned perf-smoke tooling or runs paired-reference performance bisection. Use when measuring a commit, comparing good and bad revisions, diagnosing performance regressions, or interpreting bisection artifacts.
audience: developer
status: experimental
owners:
  - isaaclab-maintainers
---

# Performance Bisection

## When To Use

Use this skill to choose and sequence the performance Skills:

- `isaaclab-perf-benchmark-commit`: collect data for one revision.
- `isaaclab-perf-threshold-check`: classify existing data without running simulation.
- `isaaclab-perf-bisect-range`: qualify endpoints and locate a first-bad commit.

Single-commit benchmarking is useful on local workstations. Full bisection can
also run locally, but recommend stable dedicated hardware when the result needs
to be authoritative. Do not assume access to internal infrastructure or automate
host provisioning.

## Workflow

1. Identify the commit or good/bad range, task, backend, workload size, metric,
   and expected GPU.
2. If the backend is genuinely undecided, hand off to the pinned
   `isaaclab-selecting-backends` Skill, then persist the choice in the plan.
3. Use `isaaclab-installing-isaac-lab` only for initial host onboarding or a
   current checkout needed by `local-source`. Never invoke it inside
   `local-reconstruct` or `docker-reconstruct`; historical reconstruction remains
   owned by deterministic runner code.
4. Run preflight and inspect `preflight.json` plus `hardware_context.json`.
   Hardware mismatch is advisory; preserve it in the report.
5. Route host or current-checkout setup failures to
   `isaaclab-setup-troubleshooting`. Keep candidate-specific skips and
   `perf_smoke_tooling_incompatible` inside the bisection evidence instead.
6. For one commit, route to `isaaclab-perf-benchmark-commit`. Resolve one
   maintained perf-smoke tooling commit SHA and use its driver and result
   contract. Use `WORKTREE` only for explicitly non-authoritative development.
7. If benchmark artifacts already exist, route deterministic classification to
   `isaaclab-perf-threshold-check`.
8. For a range, route to `isaaclab-perf-bisect-range`. Use the same pinned
   tooling, task, metric, and warmup contract for every endpoint and candidate.
9. Inspect `measurement_summary.json` for a single commit or `report.md` and
   `summary.json` for a range.
10. Treat categorized skips as evidence. Fix host/environment blockers, but do
   not fall back to benchmark code from an incompatible historical commit.
11. If a serious regression does not reproduce locally, copy
   `plan.resolved.json` to a stable equivalent GPU host and rerun the
   command recorded in `relaunch.json`.
12. After a deterministic culprit is identified, use `profile-isaac-sim` only
    when deeper profiling is requested and its release-build prerequisites apply.
    Profiling may explain the verdict but must not rewrite it.

## Validation

1. Confirm `tooling_manifest.json` records a tooling SHA, bundle hash, and contract hash.
2. Confirm every measured attempt has matching `tooling.json` and
   `tooling_verification.json` hashes plus a numeric canonical metric.
3. Confirm exactly one successful warmup per commit is recorded, shares the
   stack-scoped cache identity, and is excluded from statistics.
4. Confirm every range attempt uses the plan's one `tooling_spec_hash`.
5. Confirm hardware identity and mismatch warnings are present.
6. Run `isaaclab-bisect-upstream-skills validate` when changing upstream
   handoffs or pins.
7. For a regression fix, verify its regression test fails without the fix.

For skill changes, run:

```bash
./isaaclab.sh -p -m pytest -q tools/perf_bisection/tests
uv run --no-project python tools/skills/cli.py check
```

For bisection changes, run the focused tests in the reference before a real GPU run.

## Maintenance

Keep this skill synchronized with `tools/perf_bisection/README.md`,
`tools/perf_bisection/src/isaaclab_bisection/cli.py`,
`tools/perf_bisection/src/isaaclab_bisection/skill_api.py`,
`tools/perf_bisection/src/isaaclab_bisection/upstream_skills.lock.json`,
`tools/perf_bisection/src/isaaclab_bisection/bisection/tooling.py`, and
`tools/perf_bisection/src/isaaclab_bisection/bisection/measurement.py`. Update the maintained tooling
snapshot deliberately when its contract changes; never discover or guess a
candidate-native benchmark workflow.
Treat `perf_smoke_tooling_incompatible` as a terminal support-window result, not
as a hole to skip during binary search.

## References

- [Reference](reference.md)
- [Evaluations](evaluations.md)
- [Skill catalog](../../../skills/README.md)
- [Optional upstream Skills](../../../tools/perf_bisection/docs/upstream-skills.md)
- [Bisection README](../../../tools/perf_bisection/README.md)
- [Harness](../../../tools/perf_bisection/src/isaaclab_bisection/cli.py)
