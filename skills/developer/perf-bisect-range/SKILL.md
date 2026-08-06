---
name: isaaclab-perf-bisect-range
description: Runs paired-reference binary search across an Isaac Lab commit range with pinned tooling and a versioned JSON contract. Use when good and bad revisions reproduce a performance step and Fanes Agent or other automation must locate the first bad commit.
audience: developer
status: experimental
owners:
  - isaaclab-maintainers
---

# Bisect a Performance Regression

## When To Use

Use this skill after a good and bad revision are known or strongly suspected.
The workflow qualifies both endpoints before binary search and returns
`inconclusive` when the regression is not locally reproduced. Use
`isaaclab-perf-benchmark-commit` for one revision and
`isaaclab-perf-threshold-check` for comparison without execution.

## Workflow

1. Create an input matching [input.schema.json](input.schema.json). Use a full
   `tooling_ref` SHA and one fixed task, metric, and runner contract.
2. Run:

   ```bash
   isaaclab-bisect-skill \
       --input bisect-input.json \
       --output bisect-output.json
   ```

3. Read [output.schema.json](output.schema.json), then inspect `report.md` and
   `summary.json` through the returned artifact paths.
4. Accept a culprit only when `status` is `completed` and
   `result.suspected_first_bad_commit` is populated.
5. Hand the culprit and `result.stack_diff` to diagnosis. Do not diagnose from
   skipped or unsupported commits.

## Validation

- Confirm good/bad endpoint separation exceeds the effective noise-aware threshold.
- Confirm every attempt uses one tooling contract hash and hardware identity.
- Confirm warmup samples are excluded from statistics.
- Treat `unsupported_tooling_contract`, host blockers, and noisy endpoints as terminal evidence.

## Maintenance

Keep this skill synchronized with
`tools/perf_bisection/src/isaaclab_bisection/skill_api.py`,
`tools/perf_bisection/src/isaaclab_bisection/cli.py`,
`tools/perf_bisection/src/isaaclab_bisection/bisection/engine.py`, and
`tools/perf_bisection/src/isaaclab_bisection/bisection/paired_reference.py`. Preserve the output
envelope while evolving the embedded bisection summary additively.

## References

- [Evaluations](evaluations.md)
- [Range invocation reference](reference.md)
- [Input schema](input.schema.json)
- [Output schema](output.schema.json)
- [Bisection README](../../../tools/perf_bisection/README.md)
