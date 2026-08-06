<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Draft agent card

This is a repository draft. NVIDIA owners must transfer it into the current
authoritative PLC agent-card template before deployment.

## Identity and purpose

- Name: IsaacLab Bisection Agent
- Version: Isaac Lab source commit and package version
- Owner: Isaac Lab maintainers recorded in PLC
- Purpose: reconstruct historical IsaacLab commits, collect reproducible
  performance metrics, qualify endpoint separation, locate the first bad commit,
  and produce an evidence handoff.
- Non-purpose: autonomous source modification, issue updates, merges, releases,
  production actions, or LLM-authored performance verdicts.

## Capabilities

- Read Git history and source from one operator-selected repository.
- Create isolated local clones, Python environments, caches, and artifacts.
- Execute historical package installation and benchmark workloads.
- Start Docker containers with GPU access.
- Download pinned Python, system, Isaac Sim, and modular-stack packages.
- Optionally send bounded, redacted setup/recovery context to one explicitly
  selected OpenAI-compatible endpoint.
- Print optional installation commands for commit-pinned NVIDIA upstream Skills.

## Tools and integrations

- Local executables: Git, Docker, uv, Python, package installers, IsaacLab
  launcher, GPU runtime tools, and exact allowlisted diagnostics.
- External services: package registries and, only when enabled, the configured
  LLM endpoint.
- MCP servers: none.
- Browser/computer use: none.
- SSH: none.
- Shared-resource write APIs: none.
- Persistent model memory: none.

## Data and credentials

- Typical data: public source, Git metadata, benchmark plans, stack pins,
  subprocess logs, GPU/host diagnostics, metrics, and diffs.
- NVIDIA data: only classifications permitted by the portal and only with an
  approved endpoint/infrastructure.
- Regulated data: prohibited absent case-by-case Legal approval.
- LLM credential: explicit operator-selected environment variable, scoped and
  short-lived; never forwarded to candidate subprocesses or Docker.
- Candidate credentials: none by design.

## Autonomy and authority

- Default path: deterministic, locally invoked, bounded by a finite commit range,
  sample count, timeout, and retry count.
- Optional LLM path: recommends only predefined recovery/probe actions; security
  enforcement and verdicts remain deterministic code.
- Source-of-truth authority: none. All pushes, merges, tags, publications,
  governance changes, and report sharing require human action.

## Trust assumptions and known limitations

- Target repositories and commits are trusted code. Use approved strong
  isolation for hostile or externally controlled source.
- Docker/GPU isolation does not defend against kernel, driver, firmware, or
  container-runtime escape.
- Reconstruction egress is not domain-allowlisted by the package.
- Local modes share the operator's host filesystem identity.

## Security evidence

- Threat model: `docs/security-threat-model.md`
- Test plan: `docs/security-test-plan.md`
- Readiness/gates: `docs/security-readiness.md`
- Operational policy and incident response: `SECURITY.md`
- Compatibility and release policy: `docs/compatibility.md`,
  `docs/releasing.md`
