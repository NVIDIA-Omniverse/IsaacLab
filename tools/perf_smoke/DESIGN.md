# Perf Smoke Gate — Architecture

This document explains *how* the gate is put together: the data flow, the
baseline/history model, and the verdict logic. For a plain-English file-by-file
tour, see [`README.md`](README.md).

## Design goals

1. **Cheap and per-PR.** A handful of short, stable tasks on a fixed GPU — fast
   enough to run on every pull request.
2. **Robust to noise.** Small run-to-run wobble must not flake the gate; only a
   real, sustained drop blocks.
3. **In-tree and reviewed.** Baselines and history live in the repo and only
   change through a normal, reviewed PR — never a silent side-write from CI. This
   keeps the gate auditable and lets `git blame` answer "why did the bar move?".
4. **No new dependencies.** Everything is standard-library Python so the
   comparator and its tests run on any runner without Isaac Sim or a GPU.

## Components

| Layer | Module | Responsibility |
|---|---|---|
| Orchestration | `run_perf_gate.py` | Per task: resolve launch config from `baseline.json`, launch the benchmark as its own Isaac Sim subprocess (retry once), hand the result to the comparator, aggregate verdicts (worst wins). |
| Decision (pure logic) | `check_perf_regression.py` | Read one benchmark result + baseline + rolling window, compute the KPI, and return PASS / WARN / BLOCK. No GPU, no Isaac Sim. |
| Stored state | `baseline.json`, `perf_history/`, `baseline_overrides.json` | The launch config + static fallback, the rolling window of recent samples, and manual threshold overrides. |
| Maintenance | `rebaseline.py`, `seed_history.py` | Produce/refresh the stored state from fresh or existing runs (always via a reviewed PR). |

## Data flow (one PR)

```
baseline.json ─┐
               ├─► run_perf_gate ─► benchmark_non_rl.py (subprocess) ─► result.json
perf_history/ ─┤                                                            │
overrides ─────┘                                                            ▼
                                   check_perf_regression  ◄─────────────────┘
                                            │
                                   RESULT=PASS|WARN|BLOCK  +  $GITHUB_STEP_SUMMARY table
```

The orchestrator only builds commands and aggregates; **all** of the regression
judgement lives in the comparator, which is why the comparator is independently
unit-testable without hardware.

## The KPI

The gating metric is the **post-warm-up steady FPS** (`steady_fps`): the
benchmark's per-frame effective-FPS array with the first `warmup_frames` dropped.
Using the same statistic the backend already reports — just windowed — keeps the
measured value directly comparable to the stored history. Wall-clock seconds are
carried as a secondary, advisory signal only.

## Baseline & history model

There are two stores, deliberately layered:

- **Rolling window (`perf_history/`, primary).** Per `(task, GPU)`, the last
  *N* known-good samples. The comparator computes its threshold *at test time*
  from this window with a robust **median + MAD** estimator:

  ```
  center = median(window)
  spread = max(1.4826 * MAD(window), min_spread_pct/100 * center)
  WARN   when measured < center - k_warn  * spread
  BLOCK  when measured < center - k_block * spread
  ```

  A `min_spread_pct` floor stops a very low-variance task from blocking on
  trivial dips.

- **Static fallback (`baseline.json`, secondary).** When the window is too small
  to trust (`< MIN_WINDOW` samples), the comparator falls back to a static
  `baseline_fps` + percentage bands calibrated for that task/GPU. This keeps a
  fresh store from silently passing everything before it has accumulated history.

**Overrides** (`baseline_overrides.json`) are a manual escape hatch keyed by
*stable* test identity (`task` + GPU), applied on top of either source — used for
one-off threshold relaxations or `skip` that ride along in the PR.

### Environment fingerprint buckets

Performance is only comparable within the same software stack: a Warp bump or an
Isaac Sim upgrade can legitimately shift FPS, and mixing those samples into one
window would corrupt the baseline. So history is **bucketed by an environment
fingerprint**:

```
perf_history/
  <task>__<gpu>.json                     # flat "default" bucket (legacy / no provenance)
  env-<hash>/<task>__<gpu>.json          # one bucket per (warp, isaaclab, cuda) stack
```

- `env_fingerprint(result)` hashes the environment-defining provenance
  (`warp`, `isaaclab`, `cuda`) into a short, stable `env-<hash>` key. GPU is
  *not* in the hash because it is already in the file name.
- The comparator derives the fingerprint from the run under test and reads the
  matching bucket, **falling back to the flat file** when no bucket exists yet —
  so the change is backward-compatible with already-seeded flat history.
- A consistent filename (`history_basename`) is shared by the reader and every
  writer so a written bucket is always found again.

### Per-sample provenance

Every stored sample carries the context needed to audit or re-bucket it without
re-running the benchmark: `commit`, `warp`, `isaaclab`, `cuda`, plus the
`fingerprint` recorded at the window level. This makes the in-tree history
self-describing — a reviewer reading a `perf_history/` diff can see exactly which
commit and stack produced each number.

## Re-baselining lifecycle

`rebaseline.py` is the only writer of the stored state and serves two jobs from
one measurement path:

1. **Variance study (default).** Run each task `--repeat` times and report robust
   stats (median / CV / MAD / min / max) so thresholds can be justified to
   reviewers.
2. **Rolling re-baseline (`--apply`).** Append the new samples to the window
   (pruned to a cap, stamped with provenance, written into the env bucket) and
   refresh the static fallback in `baseline.json`.

A **boiling-frog guard** keeps a rolling baseline from quietly absorbing a real
regression: a task whose new median drops the baseline by more than
`--soft-drop-pct` is *flagged for review*; a drop beyond `--hard-drop-pct` is
*refused* (old value kept) unless `--force`. Both stores then change only through
the PR that `perf-rebaseline.yml` opens.

## CI wiring

- `perf-gate.yml` — on a PR it runs a fast, GPU-free **unit-test job** (comparator,
  orchestrator, rebaseline, fingerprint helpers) for early signal, then a
  per-task GPU matrix on the L40S fleet that posts the verdict back. Advisory
  (`continue-on-error`) until cross-runner variance is confirmed.
- `perf-rebaseline.yml` — manual workflow that runs `rebaseline.py --apply` on the
  fleet and opens a PR with the `baseline.json` + `perf_history/` diff.

## Why these boundaries

- **Pure-logic comparator** ⇒ the regression rules are fully testable on any
  runner; the GPU job only *produces* numbers, it never *decides*.
- **In-tree, reviewed state** ⇒ no opaque external baseline service; every change
  to the bar is a diff someone approved.
- **Layered window → static → override** ⇒ robust thresholds when history exists,
  a safe floor when it doesn't, and a human escape hatch when neither fits.
