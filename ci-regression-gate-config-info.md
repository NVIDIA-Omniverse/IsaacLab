# IsaacLab Perf Regression Gate: Run Configuration & Test Logic
- **Status:** Proposed (under tuning/evaluation on test runners; not yet deployed in CI)
- **Date:** 2026-06-03

> This document records the decisions and empirical reasoning based on Isaac Lab 3.0.0 performance on a single L40S GPU for how a gate run is configured and judged.

---

## Context

The gate is a pre-merge check on public `isaac-sim/IsaacLab` PRs, running on self-hosted L40S GPU runners. It is meant to be a lightweight check for two classes of errors:
1. **Performance regressions** — Large changes in KPIs (FPS-mean, wall clock time) against historical baselines
2. **Environment breaks** — import/dependency/Python-version failures, OOM, hangs

Every parameter below is chosen to **maximize signal-to-noise per unit of CI wall-clock time** to reduce cost from runners, avoid blocking PRs. The final numbers come from 5×-repeated back-to-back warm runs on a single L40S.

---

## Decisions

### D1 — Execution model: warm cache, fresh subprocess per task, run-once + retry-once
- **All runs use a WARM disk cache.** We measure steady-state compute, not asset/JIT loading, so extra cost from cold starts is not good.
- **Each (task, preset) runs in its own fresh Python subprocess**, shelled out by pytest. (Isaac Sim is a process singleton → exactly one task per process; pytest never launches the app in-process.)cccccdejterencvrncltnlcuntbdjkhjjibelggeilnn
- **Each task runs once, retried once on failure.** Retry is a WARN-level signal, not a silent pass.

**Why:** isolation + reproducibility for bisection; warm cache removes the dominant variance source (first-touch / JIT pollution, see D6).

### D2 — Camera resolution: **64 px** (no sweep)
- **Reasoning:** KPI (effective FPS) barely moves from 64→128 (a few %), then drops ~20% at 256. Warm wall-time per run climbs steeply at 256. Scaling is dominated by upstream deps/hardware, not IsaacLab logic.
- **No sweep:** points between resolutions are highly correlated; one mid-low point captures the signal cheaply.

### D3 — Environment size: **512 envs** (cartpole: **4096**)
- **Reasoning:** chosen to sit in the **mid-range** where throughput reflects IsaacLab itself (manager loop, env step logic, view tensors) — *not* dominated by dispatch overhead (too few envs) and *not* bound by GPU memory / rendering (too many). Per-env throughput is ~flat across the measured range.
- **No sweep:** performance between env-size points is highly correlated.
- **Cartpole is the exception at 4096** — it is cheap enough that the larger size keeps it in its representative regime.

### D4 — Number of frames: **300**
- **Reasoning:** warm-up and pollution events occur mostly within the first 100 frames; by 300–500 frames they account for **<10% of wall-clock** and steady-state behavior is clearly apparent. 300 is cheaper.

### D5 — Task set: **4 core tasks** (PR gate), 3 excluded
**Core (kept) — selected for stability + coverage diversity, with measured inter-run noise:**
| Task | Role | CV% | Deviation range |
|---|---|---|---|
| `Isaac-Cartpole-Direct-v0` | Classic, direct, most stable canary | 0.9% | ~2% |
| `Isaac-Factory-GearMesh-Direct-v0` | Contact-rich manipulation, many meshes | 1.3% | ~3% |
| `Isaac-Repose-Cube-Shadow-Vision-Direct-v0` | Only task with **camera enabled by default** | 1.5% | ~3.6% |
| `Isaac-Velocity-Flat-G1-v0` | Locomotion | 1.7% | ~3.5% |

**Excluded (redundant / less reliable):**
- `Isaac-Dexsuite-Kuka-Allegro-Lift-v0` — redundant with shadow-hand (no default-camera coverage); less reliable/supported across backends & versions.
- `Isaac-Velocity-Rough-Anymal-C-v0`, `Isaac-Velocity-Rough-G1-v0` — redundant locomotion tasks vs g1-flat.

**Why:** all 4 core tasks have CV < 2% (tight enough for thresholding); together they cover classic / contact-rich / vision / locomotion without paying for correlated, noisier, or less-portable tasks. Cutting 3 tasks reduces test run time cost.

### D6 — Pollution filtering (what's excluded vs kept in KPIs)
**Excluded from KPIs (one-time, start-of-run pollution), recorded separately:**
- **First 2 frames of every task** — ~2 orders of magnitude above steady-state step time (first-touch overhead). Measurement starts at frame 2.
- **Frames 2–5 (Newton) / 30–60 (PhysX) for shadow-hand** — ~2 orders of magnitude, from JIT.

**Kept in KPIs (genuine recurring task behavior, not pollution):**
- g1-flat per-episode spike from first-ground-contact.
- cartpole recurring environment-reset overhead.

**Why:** the filter removes *amortized startup artifacts* that would mask real regressions in steady-state characteristics which are more relevant for Isaac Lab's long-running tasks in its common use-case, while preserving *behavioral* costs that a regression could legitimately move.

### D7 — Timing / cost budget (analog for CI cost)
Representative single-L40S numbers (vary with queue, parallel dispatch, container spin-up):
- **Per full-gate run: ~15–20 min.** 4 core tasks ≈ **11 min** (~647 s, all backend combos); 7 tasks ≈ **14 min** (~826 s).
- **Cold-start penalty after build: 3–5 min** (Newton JIT, renderer cache, CUDA cache) — the motivation for the warm-cache requirement (D1) and the caching open item (below).
- Per-task subtotals (all combos): Cartpole ~55 s, Factory ~193 s, G1-Flat ~122 s, Shadow-Vision ~276 s; excluded: G1-Rough ~60 s, Anymal-Rough ~52 s, Dexsuite-Lift ~67 s.
- Additional costs: CI queue (variable, out of scope), container spin-up/image build (non-trivial), orchestrator/oracle (negligible), retry-on-failure (variable).

---

## Test Logic (how a run is judged)

### Verdict: `{BLOCK, WARN, PASS}`
- **BLOCK** — hard failures: dependency breakage (MR 341 `tomllib`), large KPI regression (PR 5265 47% FPS drop), malformed test.
- **WARN** — retry-on-failure, medium KPI regression.
- **PASS** — within threshold.

### KPIs (gate signals)
- **FPS-mean** (primary)
- **Wall-clock time**

### Debug information captured (not gating, for triage/bisection)
- Provenance: runner hardware/driver, dependency map, install versions.
- Debug/warning/error logs; memory; frame-time mean & median.
- Outlier accounting: count, index, magnitude of steps > 2× step-time median.

### Threshold & baseline strategy
- **Computed at test time, per-task, over a rolling window of historical data.**
- **Median + MAD** threshold; tune `k` to separate PASS / WARN / BLOCK.
- **Historical data storage:** statistics in an **orphan branch**, hashed by git subtree + dependencies. **Manual overrides** via a config file committed *with the commit* (not in the orphan branch).

**Why median+MAD:** robust to the outliers we intentionally *keep* (D6) and to the few-% inter-run noise (D5) — avoids flapping that a mean/std gate would suffer.

---

## Consequences
- The gate is **deliberately narrow and fixed** (4 core tasks, single resolution/env-size/frame-count, no sweeps) — cheap, low-variance, reproducible — at the cost of not exploring scaling behavior (judged redundant given high cross-point correlation).
- Results are **hardware-specific** (pin runner type = L40S) and **threshold-blind at the runner layer** (evaluation is a separate input), keeping the runner reusable by the future bisection agent.
- Robust stats + warm cache + pollution filtering target a **low false-positive rate** while preserving diagnostic information, the key adoption requirement for a pre-merge gate.
