# perf-baselines — performance-gate baseline store

This is an **orphan branch** (no shared history with code branches). It holds *only*
the rolling performance baselines that the CI performance-regression gate compares
against. Do not merge it into a code branch and do not add source here.

## Layout

```
<gpu_model>/<task_id>/<backend>/[<fingerprint>/]
    ├── stats.json      # {median_fps, mad_fps, sample_count, k_warn, k_block}
    └── window.ndjson   # capped FIFO of the most recent FPS samples (one per line)
```

`<fingerprint>` is `{backend_version}/{runtime_hash}/{code_fingerprint}`; an empty
fingerprint maps to the flat `<gpu>/<task>/<backend>/` bucket. Baseline *loads* relax
outward to looser buckets; *writes* target the exact fingerprint.

## Who writes here

`tools/perf_regression_gate/baseline_manager.py` appends a sample for each PASS/WARN
task (never BLOCK / HARD_FAILURE) via a temporary git worktree, using a bounded
`fetch -> rebase -> push` retry loop so concurrent runs cannot lose a sample.

Managed automatically — edit by hand only to seed or reset a bucket.
