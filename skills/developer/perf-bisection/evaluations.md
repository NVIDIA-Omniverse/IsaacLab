# Performance Bisection Evaluations

## Contents

- [Scenario 1: Commit Near The Maintained Window](#scenario-1-commit-near-the-maintained-window)
- [Scenario 2: Local Full Bisection](#scenario-2-local-full-bisection)
- [Scenario 3: Serious Regression](#scenario-3-serious-regression)
- [Scenario 4: Unsupported Historical Commit](#scenario-4-unsupported-historical-commit)
- [Scenario 5: Candidate Deletes Benchmark Code](#scenario-5-candidate-deletes-benchmark-code)
- [Scenario 6: Resource Regression](#scenario-6-resource-regression)
- [Scenario 7: Historical Reconstruction](#scenario-7-historical-reconstruction)
- [Scenario 8: Post-Bisection Profiling](#scenario-8-post-bisection-profiling)

## Scenario 1: Commit Near The Maintained Window

Query: "Benchmark this Isaac Lab commit from last week on my workstation."

Expected behavior:

- Uses `benchmark-commit`.
- Pins and records the maintained perf-smoke tooling snapshot.
- Reconstructs the commit's pinned stack and runs the external tooling driver.

Known failure modes:

- Invokes `perf_runtime.py` from the candidate checkout.
- Falls back to a historical benchmark workflow when the fixed driver fails.

## Scenario 2: Local Full Bisection

Query: "I still want to run the entire bisection on my local GPU."

Expected behavior:

- Runs `bisect-range` locally without requiring internal infrastructure.
- Records hardware mismatch/noise warnings as advisory evidence.
- Reports a result only if good/bad references reproduce and are comparable.

Known failure modes:

- Refuses local bisection merely because the host is not dedicated.
- Silently treats the local GPU as matching the target hardware.

## Scenario 3: Serious Regression

Query: "This regression needs an authoritative bisection matching CI hardware."

Expected behavior:

- Recommends a stable equivalent GPU host.
- Reuses `plan.resolved.json` and `relaunch.json`.
- Runs the same command and measurement code on that host.

Known failure modes:

- Attempts to provision or SSH to infrastructure without user direction.
- Implements a separate dedicated-host bisection path.

## Scenario 4: Unsupported Historical Commit

Query: "Bisect a range spanning the old JSON benchmark and the new RuntimeBundle benchmark."

Expected behavior:

- Runs only the pinned maintained RuntimeBundle tooling.
- Stops cleanly when an endpoint lacks the required IsaacLab API.
- Labels the terminal outcome `perf_smoke_tooling_incompatible`.
- Does not step around the unsupported commit as a binary-search hole.

Known failure modes:

- Compares metrics produced by different rulers without disclosure.
- Guesses an adapter for an unknown output schema.

## Scenario 5: Candidate Deletes Benchmark Code

Query: "This candidate deleted its benchmark scripts; can it still be measured?"

Expected behavior:

- Uses the read-only run-scoped `perf_runtime.py` snapshot.
- Records the same tooling and contract hashes as every other candidate.
- Does not inspect or invoke candidate-native benchmark entry points.

Known failure modes:

- Attempts benchmark discovery in the candidate checkout.
- Changes the result parser or metric semantics for this candidate.

## Scenario 6: Resource Regression

Query: "Bisect a GPU-memory, CPU-utilization, or host-RAM increase instead of an FPS decrease."

Expected behavior:

- Selects `runtime_resources.gpu_mem_used_mb` as the numeric metric.
- Selects `runtime_resources.cpu_util_pct` or
  `runtime_resources.system_ram_peak_mb` for canonical host resources.
- Uses `increase` as the regression direction.
- Pins that metric selection in the tooling contract for every attempt.

Known failure modes:

- Assumes every regression is an FPS decrease.
- Reads an unprojected raw-bundle field without defining canonical semantics.

## Scenario 7: Historical Reconstruction

Query: "Use the official installation Skill to set up every commit in this range."

Expected behavior:

- Refuses to put the installation Skill inside the candidate loop.
- Keeps `docker-reconstruct` responsible for each commit's pinned stack.
- Uses the installation Skill only if the operator needs current-host onboarding.

Known failure modes:

- Applies current installation docs to every historical commit.
- Adds confirmation prompts or mutable agent behavior to candidate execution.

## Scenario 8: Post-Bisection Profiling

Query: "The agent found the first bad commit. Can it profile why?"

Expected behavior:

- Preserves the completed deterministic verdict and evidence.
- Offers `profile-isaac-sim` when its release-build prerequisites match.
- Treats profiling output as diagnosis, not reclassification.

Known failure modes:

- Runs profiling before endpoint qualification or binary search completes.
- Changes the first-bad verdict based on an LLM or profiling hypothesis.
