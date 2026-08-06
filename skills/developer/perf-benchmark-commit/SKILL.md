---
name: isaaclab-perf-benchmark-commit
description: Reconstructs and benchmarks one Isaac Lab commit with pinned perf-smoke tooling and a versioned JSON contract. Use when measuring a historical revision, collecting a canonical metric, or invoking single-commit benchmarking from Fanes Agent or other automation.
audience: developer
status: experimental
owners:
  - isaaclab-maintainers
---

# Benchmark One Commit

## When To Use

Use this skill to measure one commit without comparing it to a baseline or
searching a range. For a gate verdict use `isaaclab-perf-threshold-check`; for a
first-bad search use `isaaclab-perf-bisect-range`.

## Workflow

1. Create an input matching [input.schema.json](input.schema.json). Pin
   `tooling_ref` to a full commit SHA for authoritative runs.
2. Use `local-reconstruct` for host isolation or `docker-reconstruct` with an
   explicit image for container isolation.
3. Run the JSON adapter:

   ```bash
   isaaclab-bisect-skill \
       --input benchmark-input.json \
       --output benchmark-output.json
   ```

4. Read the output using [output.schema.json](output.schema.json). Treat
   `result.succeeded: false` and categorized skips as evidence, not measured
   regressions.

The adapter streams benchmark logs to the caller and writes the output envelope
even when the harness returns a nonzero status.

## Validation

- Require `tooling_manifest.json` and matching tooling hashes for authoritative results.
- Require a numeric selected metric in every measured attempt.
- Confirm hardware identity and mismatch warnings are retained.
- Validate the request and response against the linked schemas.

## Maintenance

Keep this skill synchronized with
`tools/perf_bisection/src/isaaclab_bisection/skill_api.py`,
`tools/perf_bisection/src/isaaclab_bisection/cli.py`, and
`tools/perf_bisection/src/isaaclab_bisection/bisection/measurement.py`. Add fields additively; bump
the JSON `schema_version` before making an incompatible contract change.

## References

- [Evaluations](evaluations.md)
- [Invocation reference](reference.md)
- [Input schema](input.schema.json)
- [Output schema](output.schema.json)
- [Bisection README](../../../tools/perf_bisection/README.md)
