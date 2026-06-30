# Skill: bisection

Automates "which commit introduced this performance regression?" using binary search
over the IsaacLab commit history with LLM-driven diagnosis.

---

## CLI Invocation

```bash
python /path/to/bisect_agent/bisect.py \
    --good <good_sha> \
    --bad  <bad_sha>  \
    --task <task_id>  \
    --backend <backend_key>
```

**Minimal required flags:** `--good`, `--bad`, `--task`, `--backend`.

---

## All Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--good` | str | required | Known-good commit SHA. Performance is acceptable at this commit. |
| `--bad` | str | required | Known-bad commit SHA. Performance has regressed at this commit. |
| `--task` | str | required | IsaacLab task identifier, e.g. `Isaac-Velocity-Flat-G1-Direct`. |
| `--backend` | str | required | Simulation backend key, e.g. `newton`. |
| `--repo` | path | `../IsaacLab` | Path to the local IsaacLab git repository. Used by the commit enumerator and diff fetcher. |
| `--output-dir` | path | `runs/<task>_<backend>_<good[:7]>_<bad[:7]>` | Root directory for all run artifacts. Will be created if it does not exist. |
| `--dev` | flag | off | Dev mode: skip Docker, run stub benchmarks directly. Requires `stub_benchmark.py` at `IsaacLab/tools/perf_smoke_test/dev/stub_benchmark.py`. |
| `--dev-perf-map` | path | none | Path to a JSON file mapping SHA strings to simulated fps_mean values. Used only with `--dev`. SHAs not in the map default to 200.0 fps. |
| `--no-llm` | flag | off | Skip the orchestrator LLM and run grounding + bisect deterministically. Writes `bisect_result.json` but does NOT produce `report/diagnosis.json`. |
| `--model` | str | `$BISECT_LLM_MODEL` or `claude-sonnet-4-6` | LLM model identifier to use for orchestrator and diagnostician sessions. |
| `--base-url` | str | `$BISECT_LLM_BASE_URL` or `https://api.anthropic.com/v1` | Base URL for the OpenAI-compatible completions endpoint. Set this to use a local model (vLLM, NIM, Ollama) or a non-Anthropic provider. |

### Environment Variable Equivalents

The following env vars are read when the corresponding CLI flags are not provided:

| Env var | CLI flag | Notes |
|---------|----------|-------|
| `BISECT_LLM_MODEL` | `--model` | |
| `BISECT_LLM_BASE_URL` | `--base-url` | |
| `BISECT_LLM_API_KEY` | — | API key. Falls back to `ANTHROPIC_API_KEY`. |
| `ANTHROPIC_API_KEY` | — | Fallback API key. |

---

## Output Directory Structure

All artifacts are written under `--output-dir` (default: `runs/<task>_<backend>_<short_good>_<short_bad>/`):

```
<output-dir>/
├── run_config.json            # Inputs captured at launch time
├── status.json                # Live phase/progress (poll this)
├── grounding/
│   ├── result.json            # Baseline stats, separation verdict
│   ├── <sha>_0/               # Per-run artifacts from grounding
│   └── <sha>_1/
├── commits.json               # Ordered commit list (good-exclusive, bad-inclusive)
├── bisect/
│   ├── state.json             # Current lo/hi/tested — enables resume
│   └── <sha>/                 # Per-commit bisect run artifacts
├── bisect_result.json         # Final bisect output (first_bad_sha, etc.)
└── report/
    ├── diagnosis.json         # Root-cause diagnosis (LLM output)
    └── report.md              # Human-readable summary
```

---

## Polling `status.json`

Poll `<output-dir>/status.json` to track progress. The file is written after every
major state change by the orchestrator.

**Schema:**
```json
{
  "phase":       "grounding | enumerate | bisect | diagnosis | done | error",
  "status":      "running | complete | warn | error",
  "progress":    "<human-readable string>",
  "bisect_lo":   2,
  "bisect_hi":   6,
  "last_update": "2026-06-29T15:32:00Z"
}
```

`bisect_lo` and `bisect_hi` are only present during the `"bisect"` phase.

**Terminal states:** `phase="done"` with `status="complete"`, or `phase="error"` with
`status="error"`. Poll until one of these is reached.

**Example polling loop (bash):**
```bash
OUTPUT_DIR="runs/my_run"
while true; do
    PHASE=$(python -c "import json,sys; d=json.load(open('$OUTPUT_DIR/status.json')); print(d['phase'],d['status'])" 2>/dev/null)
    echo "$(date -u +%H:%M:%S)  $PHASE"
    if [[ "$PHASE" == "done complete" || "$PHASE" == "error error" ]]; then
        break
    fi
    sleep 30
done
```

---

## Reading Results

### `bisect_result.json`

Written when bisection converges. Present regardless of `--no-llm`.

```json
{
  "first_bad_sha":          "<sha>",
  "prev_good_sha":          "<sha or null>",
  "first_bad_message":      "<commit message>",
  "commits_tested":         7,
  "total_commits_in_range": 47,
  "kpi_deltas":             {"fps_mean": -34.9},
  "confidence":             "high | medium | low",
  "skip_count":             0
}
```

- `first_bad_sha`: the commit that introduced the regression.
- `prev_good_sha`: the last known-good commit (null if the first commit in range is bad).
- `confidence`: `"high"` if no SKIPs; `"medium"` if <3 SKIPs; `"low"` if >=3 SKIPs.

### `report/diagnosis.json`

Written by the LLM diagnostician. Only present when LLM mode is active (no `--no-llm`).

```json
{
  "first_bad_sha":    "<sha>",
  "regression_class": "upstream_dep | isaaclab_code | indeterminate",
  "kpi_impact": {
    "fps_mean": {"delta_pct": -34.9, "good": 3812.4, "bad": 2480.1}
  },
  "hypotheses": [
    {
      "id":               "H1",
      "description":      "<text>",
      "evidence":         "<text>",
      "tested":           true,
      "test_description": "<text>",
      "test_result":      "<text>",
      "conclusion":       "CONFIRMED | REFUTED | UNTESTED"
    }
  ],
  "root_cause":          "<string or null>",
  "recommended_actions": ["<action>"],
  "confidence":          "high | medium | low",
  "experiments_run":     1
}
```

- `regression_class`: high-level triage result.
  - `upstream_dep`: the regression was caused by a dependency version bump.
  - `isaaclab_code`: the regression was caused by IsaacLab source changes.
  - `indeterminate`: the diagnostician could not identify a specific cause.
- `root_cause`: a human-readable confirmed cause, or null if no hypothesis was confirmed.
- `experiments_run`: total number of benchmark re-runs the diagnostician triggered.

### `report/report.md`

Human-readable Markdown summary generated from `diagnosis.json`. Contains KPI impact
table, root cause section, hypotheses, and recommended actions. Safe to attach to a
Slack or GitHub issue.

---

## Dev Mode Usage

Dev mode runs end-to-end without Docker or a GPU. Use it for integration testing and
prompt development.

**Prerequisites:**
- `IsaacLab/tools/perf_smoke_test/dev/stub_benchmark.py` must exist.
- A SHA-to-fps_mean map must be prepared (see below).

**Build a dev perf map:**
```bash
python bisect_agent/tests/build_dev_perf_map.py \
    --repo ../IsaacLab \
    --good-count 4 \
    --bad-count  4 \
    --good-fps   3800 \
    --bad-fps    2400 \
    --output     bisect_agent/tests/dev_perf_map.json
```

This assigns simulated fps_mean values to real IsaacLab SHAs. Commits 0-3 (good range)
get 3800 fps; commits 4-7 (bad range) get 2400 fps. Bisect should identify commit 4 as
`first_bad_sha`.

**Run in dev mode:**
```bash
python bisect_agent/bisect.py \
    --good <sha_of_commit_3>     \
    --bad  <sha_of_commit_7>     \
    --task Isaac-Cartpole-Direct \
    --backend newton             \
    --dev                        \
    --dev-perf-map bisect_agent/tests/dev_perf_map.json \
    --output-dir   runs/test_e2e
```

**Verification:**
```bash
python -c "
import json
r = json.load(open('runs/test_e2e/bisect_result.json'))
print('first_bad:', r['first_bad_sha'])
print('confidence:', r['confidence'])
print('commits_tested:', r['commits_tested'])
"
```

Expected: `first_bad_sha` matches SHA of commit 4 from the map; bisect used
`ceil(log2(N)) + 2` or fewer runs.

**Skip LLM in dev mode (deterministic only):**
```bash
python bisect_agent/bisect.py \
    --good <sha> --bad <sha> \
    --task Isaac-Cartpole-Direct --backend newton \
    --dev --dev-perf-map tests/dev_perf_map.json \
    --no-llm \
    --output-dir runs/test_nollm
```

`report/diagnosis.json` will not be written. `bisect_result.json` will be present.

---

## Integration Pattern (regression-agent)

When the regression-agent identifies a regression between two CI builds:

1. Map build numbers to IsaacLab git SHAs (via build metadata or CI logs).
2. Invoke bisect.py with the SHAs.
3. Poll `status.json` until `phase="done"`.
4. Read `report/diagnosis.json` for `regression_class`, `root_cause`, and
   `recommended_actions`.
5. Include `report/report.md` in the Slack or GitHub notification.

```
regression-agent:
  "Isaac-Velocity-Flat-G1 fps dropped ~35% between build 41500 and 41600"
    -> map to SHA_A (build 41500), SHA_B (build 41600)
    -> python bisect.py --good SHA_A --bad SHA_B \
           --task Isaac-Velocity-Flat-G1-Direct --backend newton
    -> poll status.json until phase=done
    -> read report/diagnosis.json
    -> include in Slack report
```

---

## Resume Behavior

Bisect runs are fully resumable. If the process is interrupted at any point, re-run
the same command with the same `--output-dir`. Each stage checks for its output
artifact before doing any work:

- `grounding/result.json` exists → grounding is skipped.
- `commits.json` exists → enumeration is skipped.
- `bisect/state.json` exists → bisect resumes from the saved `lo`/`hi` bounds.
- `bisect_result.json` exists → bisect is skipped entirely.
- `report/diagnosis.json` exists → diagnosis is skipped.

No flags are needed to trigger resume; it is automatic.
