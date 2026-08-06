<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Draft Skill cards

These are repository drafts. NVIDIA owners must transfer each Skill into the
current authoritative PLC Skill-card format and onboard approved versions to
NV-CARPS before broad internal sharing.

## `isaaclab-perf-bisection`

- Purpose: guide an operator through deterministic commit benchmarking,
  endpoint qualification, bisection, evidence interpretation, and optional
  post-bisection handoffs.
- Inputs: target repository, refs, task/backend, runner mode, metric policy,
  sample policy, and output location.
- Capabilities: invokes local package CLIs and reads generated artifacts.
- Data access: selected repository, local logs, and run artifacts.
- Writes: local run directories only.
- Network/credentials: inherited from the selected runner; no Skill-owned
  credential.
- Restrictions: no direct shared-resource write; no LLM verdict authority;
  upstream Skills stay outside deterministic classification.

## `isaaclab-perf-benchmark-commit`

- Purpose: reconstruct and benchmark one selected commit with pinned tooling.
- Inputs: commit, task/backend, target repository, runner configuration, output.
- Capabilities: invokes the single-commit runner and package installation.
- Data access: selected repository, package metadata, benchmark output.
- Writes: local candidate/environment/cache/artifact directories.
- Network/credentials: package downloads; candidate environment excludes agent
  credentials.
- Restrictions: target code must be trusted or run on approved strong
  isolation; output is evidence, not a shared publication.

## `isaaclab-perf-bisect-range`

- Purpose: qualify good/bad references and locate the first bad commit.
- Inputs: ancestry-valid range plus benchmark and decision policy.
- Capabilities: repeatedly invokes the deterministic runner and comparator.
- Data access: selected Git range and run artifacts.
- Writes: local run state and reports only.
- Network/credentials: same as the selected runner; no Skill-owned credential.
- Restrictions: no first-bad verdict when reference qualification is
  inconclusive; no merge, issue update, or release action.

## `isaaclab-perf-threshold-check`

- Purpose: classify existing benchmark data against a versioned deterministic
  threshold contract without running simulation.
- Inputs: versioned benchmark/reference JSON and threshold policy.
- Capabilities: local parsing and deterministic comparison only.
- Data access: explicitly supplied local JSON.
- Writes: structured local response.
- Network/credentials: none.
- Restrictions: rejects malformed contracts; no source execution, LLM, Docker,
  or shared-resource write.

## Review triggers

Re-review and republish a card when changing its inputs, outputs, tools, network,
credentials, data classes, autonomy, write scope, or trust assumptions.
