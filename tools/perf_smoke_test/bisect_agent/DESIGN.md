# Bisection Agent — Design Reference

> **Purpose**: This document is both the authoritative design spec and the reference context
> for all implementation agents. Every module must implement the interfaces described here.

---

## 1. Executive Summary

The bisection agent automates "which commit introduced this regression?" Given a known-good
commit, a known-bad commit, a task (e.g. `Isaac-Velocity-Flat-G1-Direct`), and a backend
(e.g. `newton`), it:

1. **Grounds** the investigation: runs good+bad commits repeatedly to establish empirical
   baselines, assess statistical separation, and identify which KPIs actually regress.
2. **Bisects**: binary searches the commit range for the first bad commit (leftmost-BAD),
   running each candidate in an isolated environment (Docker in production, direct subprocess
   in dev mode).
3. **Diagnoses**: an LLM agent with running capability performs iterative hypothesis testing —
   code diffs, dependency changes, targeted re-runs — and reports only empirically-validated
   conclusions.

---

## 2. Architecture

```
bisect.py (CLI)
    │
    ├─ writes run_config.json
    └─ starts orchestrator.py (persistent LLM session)
            │
            ├─[tool] run_experiments(shas, task, backend, n)
            │           └─ core/runner.py → infra/container.py
            │
            ├─[tool] assess_grounding(run_dir)
            │           └─ core/grounding.py → core/verdict.py
            │
            ├─[tool] enumerate_commits(good_sha, bad_sha)
            │           └─ infra/commits.py (git or GitHub API)
            │
            ├─[tool] bisect_step(run_dir, commits)
            │           └─ core/bisector.py → core/runner.py + core/verdict.py
            │
            ├─[tool] fetch_diff(sha_a, sha_b)
            │           └─ infra/commits.py
            │
            ├─[tool] run_diagnosis(context)
            │           └─ core/diagnosis.py (separate LLM sub-session)
            │
            └─[tool] write_status(phase, msg)
                        └─ writes status.json (polled by integrating agents)
```

### Data flow (artifact-based, every stage independently resumable)

```
run_config.json
    │
    ├─► grounding/{sha}_{i}.json ──► grounding/result.json
    │           (run_result per run)       (assessed baseline)
    │
    ├─► commits.json
    │           [{sha, date, author, message, dep_files_changed}, ...]
    │
    ├─► bisect/{sha}.json ◄────────── uses grounding/result.json
    │           (run_result + bisect_verdict)
    ├─► bisect/state.json            (lo, hi, tested — enables resume)
    ├─► bisect_result.json           (first_bad_sha, prev_good_sha, ...)
    │
    └─► report/diagnosis.json
        report/report.md
```

---

## 3. Module Interfaces (AUTHORITATIVE)

### 3.1 `core/runner.py` — Execution Primitive

Shared by grounding and bisect. The ONLY module that triggers benchmark execution.

```python
def run_commit(
    sha: str,
    task_id: str,
    backend: str,
    output_dir: Path,
    *,
    dev_mode: bool = False,
    dev_perf_map: dict | None = None,  # sha -> fps_mean for dev mode
) -> dict:  # returns run_result dict (see schema 4.1)
```

- In production: delegates to `infra/container.py`
- In dev mode (`dev_mode=True`): calls `stub_benchmark.py` directly via subprocess,
  setting `STUB_FPS_MEAN` from `dev_perf_map[sha]` (or default 200.0 if not in map).
  The stub_benchmark path is `IsaacLab/tools/perf_smoke_test/dev/stub_benchmark.py`.
- Writes `perf_smoke_test_result.json` to `output_dir` via Phase 2 (build_bench_result.py).
  In dev mode, calls build_bench_result.py via subprocess as well.
- Returns a run_result dict (schema 4.1).

### 3.2 `core/verdict.py` — Statistical Analysis

No side effects. Pure functions. Imports from IsaacLab's oracle.py and gate_types.py.

```python
def compute_kpi_stats(run_results: list[dict]) -> dict:
    """
    Compute median + MAD + CV for each numeric KPI across multiple run_results.
    KPIs extracted: raw_fps_mean, raw_fps_p5, raw_fps_median, wall_time_s.
    Returns: {"fps_mean": {"median": X, "mad": Y, "cv": Z, "n": N}, ...}
    """

def check_separation(good_stats: dict, bad_stats: dict) -> dict:
    """
    Check if good and bad distributions are statistically separated per KPI.
    Separation ratio = |good_median - bad_median| / max(good_mad, bad_mad).
    Returns: {
        "separated": bool,
        "kpis_regressing": list[str],   # KPIs with separation_ratio >= 1.5
        "kpi_deltas": {kpi: relative_change_pct},
        "separation_ratios": {kpi: float},
        "note": str | None,
    }
    """

def classify_bisect_verdict(
    run_result: dict,
    good_stats: dict,
    kpis_regressing: list[str],
) -> str:
    """
    Returns "GOOD", "BAD", or "SKIP".
    - SKIP if run_result has failure_phase != null or raw_fps_mean is absent
    - BAD if mean of kpis_regressing metrics is below good_median - 1.5*good_mad
    - GOOD otherwise
    """
```

### 3.3 `core/grounding.py` — Grounding Phase

```python
def run_grounding(
    good_sha: str,
    bad_sha: str,
    task_id: str,
    backend: str,
    run_dir: Path,
    runner_run_commit: callable,  # core/runner.py::run_commit
    *,
    n_start: int = 3,
    n_max: int = 12,
    cv_threshold: float = 0.08,
) -> dict:  # grounding_result schema (4.2)
```

Algorithm:
1. Run `n_start` runs each of good_sha and bad_sha.
2. Check CV for each KPI. If any primary KPI CV > `cv_threshold`, run +3 more. Repeat up to n_max.
3. Call `verdict.check_separation(good_stats, bad_stats)`.
4. If `separated=False`: set `verdict="WARN_NO_SEPARATION"` but continue (don't abort).
5. Write `grounding/result.json` and return it.
6. Individual run artifacts go to `grounding/{sha}_{i}/`.

### 3.4 `core/bisector.py` — Binary Search

```python
def run_bisect(
    commits: list[dict],           # from commits.json, ordered good→bad (exclusive good)
    grounding_result: dict,
    run_dir: Path,
    runner_run_commit: callable,
) -> dict:  # bisect_result schema (4.4)
```

- Implements leftmost-BAD binary search (standard git bisect).
- Each step: run candidate commit, classify via `verdict.classify_bisect_verdict`.
- On SKIP: try mid+1 then mid-1 before giving up on that range.
- After finding first_bad: run it 1 more time to confirm (unless n_max skips exhausted).
- Writes `bisect/{sha}/` artifacts and `bisect/state.json` after each step.
- Writes `bisect_result.json` when done.

### 3.5 `infra/container.py` — Container/Subprocess Dispatch

```python
def run_in_container(
    sha: str,
    task_id: str,
    backend: str,
    output_dir: Path,
    isaaclab_repo_path: Path,
    *,
    dev_mode: bool = False,
    dev_perf_map: dict | None = None,
    docker_image: str = "bisect-runner:latest",
) -> dict:  # {exit_code, wall_time_s, artifact_dir: str}
```

Production mode: `docker run --gpus all -v {output_dir}:/artifacts -e COMMIT_SHA={sha} ...`
Dev mode: no Docker; calls stub_benchmark.py directly with `STUB_FPS_MEAN` from dev_perf_map.

**Dependency isolation — no venvs needed:** Each `docker run` starts from the same base
image with an ephemeral filesystem. Commit A's installed packages have zero effect on
Commit B's run — stronger isolation than a venv (full filesystem boundary per commit).
Venvs inside containers would be redundant overhead.

**Entrypoint must use `./isaaclab.sh -i`, NOT bare `pip install`:** Isaac Sim ships a
bundled Python environment with pre-installed packages (warp, physx, torch at fixed
versions). Using `pip install -e ".[all]"` directly risks silently downgrading or
corrupting bundled packages if a commit's requirements specify an older version.
`./isaaclab.sh -i` is the official installer that knows how to target the correct
Python environment and handle these conflicts. The entrypoint.sh stub reflects this
with a TODO noting it must be validated against a real Isaac Sim image before production use.

### 3.6 `infra/commits.py` — Commit Enumeration + Diff

```python
def enumerate_commits(
    good_sha: str,
    bad_sha: str,
    *,
    repo_path: Path | None = None,       # local git repo path
    github_repo: str | None = None,       # "owner/repo" for GitHub API
    github_token: str | None = None,
) -> list[dict]:  # list of commit schema (4.3); excludes good_sha, includes bad_sha

def fetch_diff(
    sha_a: str,
    sha_b: str,
    *,
    repo_path: Path | None = None,
    github_repo: str | None = None,
    github_token: str | None = None,
) -> dict:
    # Returns:
    # {
    #   "files_changed": [{"path": str, "additions": int, "deletions": int}],
    #   "dep_files_changed": list[str],  # subset of files matching requirements/setup/pyproject
    #   "dep_changes": list[str],         # lines from dep file diffs
    #   "diff_summary": str,              # full diff text (truncated at 8000 chars)
    #   "commit_message": str,
    # }
```

Default backend: `git` (uses local repo, no credentials). GitHub backend is optional.
`dep_files_changed` is populated when any of: `requirements.txt`, `setup.cfg`,
`pyproject.toml`, `requirements/*.txt` are modified.

### 3.7 `infra/llm_client.py` — LLM-Agnostic Tool Loop

```python
class LLMClient:
    def __init__(
        self,
        model: str,             # e.g. "claude-sonnet-4-6" or "gpt-4o"
        base_url: str | None,   # e.g. "https://api.anthropic.com/v1" or None for OpenAI default
        api_key: str,
    ): ...

    def run_session(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict],              # OpenAI-format tool schemas
        tool_dispatch: dict[str, callable],  # name -> sync callable
        *,
        max_tool_rounds: int = 40,
    ) -> str:
        """
        Runs the tool-use loop:
          1. Send messages to LLM
          2. If finish_reason == "tool_calls": dispatch tools, append results, repeat
          3. If finish_reason == "stop": return final text
        All tool calls are dispatched to tool_dispatch[name](**args).
        Returns the final assistant text response.
        """
```

Configuration via env vars:
- `BISECT_LLM_MODEL` (default: `claude-sonnet-4-6`)
- `BISECT_LLM_BASE_URL` (default: `https://api.anthropic.com/v1`)
- `BISECT_LLM_API_KEY` (falls back to `ANTHROPIC_API_KEY`)

### 3.8 `core/diagnosis.py` — Iterative LLM Diagnosis

```python
def run_diagnosis(
    bisect_result: dict,
    grounding_result: dict,
    run_dir: Path,
    llm_client: "LLMClient",
    commits_fetch_diff: callable,     # infra/commits.py::fetch_diff
    runner_run_commit: callable,      # core/runner.py::run_commit
) -> dict:  # diagnosis.json schema (4.5)
```

Registers tools:
- `fetch_diff(sha_a, sha_b)` → diff summary
- `run_experiment(sha, extra_env)` → run commit with modified env (e.g. pinned dep version)
- `read_artifact(path)` → read any file under run_dir

Runs a focused LLM sub-session (see `prompts/diagnostician.md`).
Writes `report/diagnosis.json` and `report/report.md`.

### 3.9 `orchestrator.py` — Persistent LLM Orchestrator

```python
def run_orchestrator(
    run_config: dict,
    run_dir: Path,
    llm_client: "LLMClient",
) -> None:
```

Registers all tools (see Section 2). Runs the main LLM loop using `prompts/orchestrator.md`
as system prompt. Writes `status.json` after each major state change.

### 3.10 `bisect.py` — CLI Entry Point

```
python bisect.py \
    --good <sha> \
    --bad <sha> \
    --task <task_id> \
    --backend <backend_key> \
    [--repo <path>]          # path to IsaacLab repo (default: ../IsaacLab)
    [--output-dir <path>]    # default: runs/{task}_{backend}_{good[:7]}_{bad[:7]}
    [--dev]                  # dev mode: no Docker, use stub benchmark
    [--dev-perf-map <json>]  # path to SHA->fps_mean map for dev mode
    [--no-llm]               # skip orchestrator LLM, run deterministic stages only
    [--model <model>]        # LLM model override
```

---

## 4. Artifact Schemas

### 4.1 `run_result.json` (per single benchmark run)

```json
{
  "sha": "abc1234",
  "task_id": "Isaac-Velocity-Flat-G1-Direct",
  "backend": "newton",
  "run_index": 0,
  "exit_code": 0,
  "wall_time_s": 187.3,
  "failure_phase": null,
  "raw_fps_mean": 3812.4,
  "raw_fps_median": 3808.1,
  "raw_fps_p5": 3741.2,
  "raw_fps_p95": 3889.7,
  "wall_time_s": 187.3,
  "gpu_mem_used_mb": 4821.0,
  "artifact_dir": "grounding/abc1234_0"
}
```

Fields extracted from `perf_smoke_test_result.json`. `failure_phase` is non-null if the run
crashed (import/init/runtime/oom/hang). KPI fields are null if failure_phase is non-null.

### 4.2 `grounding/result.json`

```json
{
  "good_sha": "abc1234",
  "bad_sha": "def5678",
  "task_id": "Isaac-Velocity-Flat-G1-Direct",
  "backend": "newton",
  "n_good": 4,
  "n_bad": 4,
  "good_stats": {
    "fps_mean": {"median": 3812.4, "mad": 45.2, "cv": 0.012, "n": 4},
    "gpu_mem_used_mb": {"median": 4821.0, "mad": 31.0, "cv": 0.006, "n": 4}
  },
  "bad_stats": {
    "fps_mean": {"median": 2480.1, "mad": 38.7, "cv": 0.016, "n": 4}
  },
  "separated": true,
  "kpis_regressing": ["fps_mean"],
  "kpi_deltas": {"fps_mean": -34.9},
  "separation_ratios": {"fps_mean": 29.7},
  "high_variance_tasks": [],
  "verdict": "PROCEED",
  "note": null
}
```

`verdict`: `"PROCEED"` | `"WARN_NO_SEPARATION"` | `"WARN_HIGH_VARIANCE"`.

### 4.3 `commits.json`

```json
[
  {
    "sha": "abc0001",
    "short_sha": "abc0001",
    "date": "2026-06-01T12:00:00Z",
    "author": "Alice",
    "message": "feat: add reward shaping",
    "dep_files_changed": false
  }
]
```

Ordered from first after `good_sha` to `bad_sha` inclusive.

### 4.4 `bisect_result.json`

```json
{
  "first_bad_sha": "abc0005",
  "prev_good_sha": "abc0004",
  "first_bad_message": "perf: optimize sim step",
  "commits_tested": 7,
  "total_commits_in_range": 47,
  "kpi_deltas": {"fps_mean": -34.9},
  "confidence": "high",
  "skip_count": 0
}
```

### 4.5 `report/diagnosis.json`

```json
{
  "first_bad_sha": "abc0005",
  "regression_class": "isaaclab_code",
  "kpi_impact": {
    "fps_mean": {"delta_pct": -34.9, "good": 3812.4, "bad": 2480.1},
    "gpu_mem_used_mb": {"delta_pct": 22.1, "good": 4821.0, "bad": 5887.0}
  },
  "hypotheses": [
    {
      "id": "H1",
      "description": "Dep bump: warp 1.3.0 introduced in this commit",
      "evidence": "requirements.txt changed warp==1.2.3 → warp==1.3.0",
      "tested": true,
      "test_description": "Ran first_bad with warp==1.2.3 pinned",
      "test_result": "fps restored to 3790 (baseline range)",
      "conclusion": "CONFIRMED: upstream warp regression"
    }
  ],
  "root_cause": "Upstream warp 1.3.0 regression (confirmed empirically)",
  "recommended_actions": [
    "Pin warp==1.2.3 in requirements.txt until upstream fixes it",
    "File issue at https://github.com/NVIDIA/warp"
  ],
  "confidence": "high",
  "experiments_run": 3
}
```

### 4.6 `status.json`

```json
{
  "phase": "bisect",
  "status": "running",
  "progress": "tested 4/7 commits",
  "bisect_lo": 2,
  "bisect_hi": 6,
  "last_update": "2026-06-29T15:32:00Z"
}
```

---

## 5. Hard Constraints

1. **runner.py is the ONLY thing that triggers benchmark execution.** No other module spawns
   containers or calls benchmark scripts directly.

2. **verdict.py has no side effects.** Pure functions only, no file I/O, no subprocess calls.

3. **Every stage checks for its output artifact before running.** If `grounding/result.json`
   exists, skip grounding. `bisect/state.json` enables mid-search resume.

4. **Diagnosis only states conclusions backed by a run.** Untested hypotheses are labeled
   `"tested": false` and never appear in `root_cause`. This is enforced by the diagnostician
   prompt, not by code.

5. **Container image is built once, reused across all commits.** No per-commit image rebuild.

6. **No IsaacLab source changes required.** bisect_agent/ is self-contained. It calls
   IsaacLab scripts by path via subprocess, does not import from them.

7. **`--dev` flag enables full end-to-end testing without GPU or Docker.** Dev mode uses
   stub_benchmark.py with STUB_FPS_MEAN from dev_perf_map. All modules must respect this flag.

8. **LLM is not required for stages 1-3.** `--no-llm` flag runs grounding + bisect
   deterministically and skips orchestrator + diagnosis.

---

## 6. Diagnosis Intelligence Specification

The diagnosis agent has more capability than any pure analytics tool because it can:
- Read actual code (git diffs)
- Run actual experiments (re-run commits with modified conditions)
- Synthesize across multiple information sources

### 6.1 Evidence Sources (all available as LLM tools)

| Tool | What it provides |
|------|-----------------|
| `fetch_diff(sha_a, sha_b)` | Files changed, dep file diffs, commit message, diff text |
| `run_experiment(sha, extra_env)` | Run commit with env overrides (e.g. pinned dep) |
| `read_artifact(path)` | Any file under run_dir (grounding results, bisect runs) |
| `get_task_kpi_context(task_id)` | Known variance characteristics for this task type |

### 6.2 Triage Protocol (encoded in diagnostician.md)

```
Step 1: Load context
  - Read bisect_result.json (first_bad_sha, kpi_deltas)
  - Read grounding/result.json (baseline stats, separation ratios)
  - Read bisect/{first_bad_sha}/ run_result.json (all KPIs)

Step 2: Fetch evidence
  - fetch_diff(prev_good_sha, first_bad_sha)
  - Examine: files_changed, dep_files_changed, dep_changes, commit_message

Step 3: Cross-KPI pattern analysis (informs hypothesis direction)
  - fps_mean↓ + gpu_mem↑        → memory pressure / allocation regression
  - fps_mean↓ + wall_time↑      → CPU-side bottleneck or serialization
  - fps_mean↓ small + high cv   → noise, inconclusive — note and proceed cautiously
  - fps_mean↓ + dep bump        → upstream dep is primary suspect

Step 4: Triage by commit content — FOUR CASES

  CASE A — Commit ONLY changed dep files (no other code):
    → Evidence is sufficient: the dep change is the cause by elimination.
    → Conclude: "upstream dependency regression — dep X bumped from A to B"
    → Confidence: "medium" (evidence-based, no confirming experiment run)
    → Recommended action: "pin X==A until upstream fixes; file issue at X's tracker"
    → STOP. Do NOT investigate further. Do NOT run experiments.

  CASE B — Commit changed BOTH dep files AND code:
    → List both as hypotheses with their evidence strings.
    → Optionally run ONE experiment if easy (e.g. re-run with old dep version pinned).
    → If experiment confirms dep → CASE A conclusion; if refutes → proceed to code.
    → If experiment is infeasible or expensive, leave both as "untested" hypotheses.
    → Confidence: "medium" if untested, "high" if confirmed.
    → STOP after at most 2 experiments.

  CASE C — Commit changed ONLY code:
    → Identify changed files. Flag any in the benchmark hot path:
      (sim step, physics step, reward/obs computation, environment reset, GPU kernels)
    → Form 1-2 specific hypotheses about which change could cause the KPI pattern.
    → ONE confirming experiment if straightforward (re-run to check, or targeted env flag).
    → Conclude: "likely code regression in {module} — {specific change}" with evidence.
    → Confidence: "high" if experiment confirms, "medium" if only diff evidence.
    → STOP after at most 2 experiments.

  CASE D — No obvious cause (merge commit, large refactor, ambiguous diff):
    → Note: "regression detected but no single clear cause in diff"
    → List top 2-3 candidate files by relevance to benchmark hot path.
    → Set confidence: "low", regression_class: "indeterminate"
    → Recommended action: manual investigation with profiler (Tracy/Nsight).
    → STOP — do NOT run experiments on unclear hypotheses. Save runs for when
      cause is at least plausible.

Step 5: Write diagnosis
  - Populate all fields in diagnosis.json schema.
  - root_cause: null if nothing confirmed; string if at least medium-confidence cause found.
  - hypotheses list: one entry per hypothesis (tested or untested).
  - experiments_run: count of run_experiment calls made.
  - Call write_diagnosis tool with the completed JSON.
```

**Experiment budget: max 3 total across all steps.** If budget exhausted, conclude
with what you have.

### 6.3 What NOT to do (constraints in diagnostician.md)

- Do NOT claim a high-confidence cause without evidence (diff or experiment)
- Do NOT run experiments on vague hypotheses — only when a specific, testable cause exists
- Do NOT investigate upstream dep source code — a version diff is sufficient evidence
- Do NOT re-run commits already in bisect/state.json — use those results instead
- Do NOT continue investigating after CASE A (dep-only commit) — it's already done
- Do NOT attribute regression to code that only changed comments, formatting, or docs

---

## 7. Test Scenario (Dev Mode End-to-End Test)

### 7.1 Artificial Regression Setup

File: `tests/dev_perf_map.json`

Maps real IsaacLab commit SHAs to simulated fps_mean values:
- Commits 0-3 (good range): fps_mean=3800
- Commit 4 (first bad): fps_mean=2400  ← bisect should find this
- Commits 5-7 (bad range): fps_mean=2400

Use the most recent 8 commits from IsaacLab git history.

### 7.2 Running the Test

```bash
# Build the perf map (uses real IsaacLab SHAs)
python tests/build_dev_perf_map.py \
    --repo ../IsaacLab \
    --good-count 4 \
    --bad-count 4 \
    --good-fps 3800 \
    --bad-fps 2400 \
    --output tests/dev_perf_map.json

# Run bisection in dev mode
python bisect.py \
    --good <sha_of_commit_3> \
    --bad <sha_of_commit_7> \
    --task Isaac-Cartpole-Direct \
    --backend newton \
    --dev \
    --dev-perf-map tests/dev_perf_map.json \
    --output-dir runs/test_e2e

# Expected: bisect_result.json shows first_bad = sha_of_commit_4
```

### 7.3 Verification Criteria

- `runs/test_e2e/grounding/result.json` exists with `separated=true`, `kpi_deltas.fps_mean ≈ -37%`
- `runs/test_e2e/bisect_result.json` exists with `first_bad_sha == sha_of_commit_4`
- `runs/test_e2e/report/diagnosis.json` exists with at least 1 hypothesis
- `runs/test_e2e/report/report.md` exists and is human-readable
- Binary search used ≤ ceil(log2(N)) + 2 runs (efficiency check)

---

## 8. Integration with regression-agent

Future flow (documented in SKILL.md):
```
regression-agent identifies regression:
  "Isaac-Velocity-Flat-G1 fps dropped ~35% between build 41500 and 41600"
→ map builds to SHAs
→ invoke bisect_agent:
    python bisect.py --good SHA_A --bad SHA_B --task Isaac-Velocity-Flat-G1-Direct --backend newton
→ poll status.json until "phase": "done"
→ read report/diagnosis.json
→ include in regression-agent Slack report
```

---

## 9. Directory Layout

```
bisect_agent/
├── bisect.py                         # CLI entry point
├── orchestrator.py                   # LLM orchestrator with tool registration
├── core/
│   ├── runner.py                     # Execution primitive (shared)
│   ├── verdict.py                    # Statistics (pure functions)
│   ├── grounding.py                  # Grounding phase
│   ├── bisector.py                   # Binary search phase
│   └── diagnosis.py                  # LLM diagnosis phase
├── infra/
│   ├── container.py                  # Docker / dev subprocess dispatch
│   ├── commits.py                    # git log / GitHub API
│   └── llm_client.py                 # OpenAI-compat tool loop
├── container/
│   ├── Dockerfile                    # FROM isaac-sim; reused across commits
│   └── entrypoint.sh                 # git checkout → pip install → Phase1 → Phase2
├── prompts/
│   ├── orchestrator.md               # System prompt: bisect protocol + decision rules
│   └── diagnostician.md             # System prompt: evidence-first analysis
├── schemas/                          # JSON schemas (documentation + optional validation)
│   ├── run_result.schema.json
│   ├── grounding_result.schema.json
│   ├── commit.schema.json
│   ├── bisect_state.schema.json
│   ├── bisect_result.schema.json
│   └── diagnosis.schema.json
├── tests/
│   ├── build_dev_perf_map.py         # Generate test fixture from real SHAs
│   ├── dev_perf_map.json             # SHA -> fps_mean for test scenario
│   └── test_e2e.py                   # Automated end-to-end test
├── .agents/skills/bisection/
│   └── SKILL.md                      # Integration interface for regression-agent
└── requirements.txt                  # openai, (docker optional)
```
