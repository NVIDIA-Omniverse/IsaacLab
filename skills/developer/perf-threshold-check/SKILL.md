---
name: isaaclab-perf-threshold-check
description: Applies Isaac Lab CI-gate or paired-reference threshold logic to existing benchmark data through a versioned JSON contract. Use when classifying a measurement without rerunning simulation, checking endpoint separation, or invoking the performance oracle from Fanes Agent or other automation.
audience: developer
status: experimental
owners:
  - isaaclab-maintainers
---

# Check Performance Thresholds

## When To Use

Use this skill when benchmark data already exists and only a deterministic
verdict is needed. It does not run Isaac Sim, reconstruct environments, or
search commits.

Choose `ci_gate` for `PASS`/`WARN`/`BLOCK`/`HARD_FAILURE` against rolling
baseline statistics and configured FPS floors. Choose `paired_reference` for
reference-signal checks or `GOOD`/`BAD`/`UNCLEAR` candidate classification.

## Workflow

1. Create an input matching [input.schema.json](input.schema.json).
2. Run:

   ```bash
   isaaclab-bisect-skill \
       --input threshold-input.json \
       --output threshold-output.json
   ```

3. Read `result` using [output.schema.json](output.schema.json).
4. Preserve the exact input with the response so the threshold decision is reproducible.

## Validation

- Confirm the metric direction matches the quantity: FPS decreases regress;
  resource use increases regress.
- For `ci_gate`, use a canonical `perf_smoke_test_result.json`.
- For paired comparison, require positive baseline medians and preserve observed spread.
- Do not reinterpret `SKIP`, `UNCLEAR`, or `HARD_FAILURE` as a measured regression.

## Maintenance

Keep this skill synchronized with
`tools/perf_bisection/src/isaaclab_bisection/skill_api.py`,
`tools/perf_bisection/src/isaaclab_bisection/oracle.py`, and
`tools/perf_bisection/src/isaaclab_bisection/bisection/paired_reference.py`. Update schemas additively
with implementation changes.

## References

- [Evaluations](evaluations.md)
- [Threshold modes](reference.md)
- [Input schema](input.schema.json)
- [Output schema](output.schema.json)
- [Module interfaces](../../../tools/perf_bisection/README.md)
