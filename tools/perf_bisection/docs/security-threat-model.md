<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Security threat model

This repository-scoped threat model is evidence for, but not a substitute for,
the formal NVIDIA SecNemo/Security Architecture TAVA required before internal
deployment.

## Assets and security objectives

- Protect host credentials, source trees, home-directory data, Docker state,
  internal network access, and GPU host availability.
- Preserve benchmark-plan and result integrity so untrusted output cannot alter
  deterministic `GOOD`/`BAD` classification.
- Prevent credentials and sensitive source or log content from leaking through
  model requests, terminal output, artifacts, images, or published packages.
- Keep every shared-resource write human-reviewed and attributable.

## Trust boundaries and data flow

1. The operator supplies a plan and a target Git repository to the CLI.
2. The deterministic engine selects commits and invokes the runner.
3. The runner snapshots pinned benchmark tooling separately from candidate
   source.
4. `docker-reconstruct` mounts the target repository read-only and uses writable,
   run-scoped candidate, environment, JIT, Kit, and artifact directories.
5. Historical package metadata, build hooks, install scripts, and benchmark code
   execute inside that reconstruction environment and can access its outbound
   network.
6. Candidate output becomes local logs and structured artifacts.
7. When explicitly enabled, an approved LLM endpoint receives a redacted subset
   of plan and log data and returns a bounded recovery or probe decision.
8. Deterministic code, never the LLM, performs metric classification and binary
   search.
9. Reports remain local drafts until a human chooses to share, commit, push,
   publish, or merge them.

## Threat analysis

### Spoofing

- A mutable target ref or tooling ref could measure different code than the
  operator intended. The engine resolves refs to commits and records tooling
  hashes in run artifacts.
- Every real execution requires a fresh `--trust_target_code` confirmation so
  automation cannot silently promote an unreviewed repository into executable
  input.
- A spoofed LLM endpoint could receive log data and return malicious decisions.
  LLM use requires an explicit absolute URL, HTTPS for non-loopback hosts, and
  rejects credentials embedded in the URL. NVIDIA deployments must additionally
  select only portal-approved endpoints.
- Container base tags could move. The Dockerfile pins both external base
  image manifests by digest.

Residual risk: TLS and registry trust remain dependencies. Release owners must
review intentional digest updates and scan the resulting image.

### Tampering

- Candidate code could modify benchmark tooling to bias results. Tooling is
  snapshotted independently, mounted read-only in Docker, hashed, and verified.
- A crafted plan could redirect cleanup or writable mounts outside the run.
  Task/backend identifiers are safe path components, tooling paths must be
  relative, run-scoped paths are resolved below the output root, symlink escape
  is rejected, and passthrough arguments cannot override protected paths.
- Model output could request a result-changing action. Recovery actions map to a
  closed vocabulary; diagnostics map to exact read-only argv; base-image repair
  package names are allowlisted; no model action can emit a performance verdict.
- A local-mode candidate can alter host-accessible files with the operator's
  identity. Local modes are explicitly restricted to trusted repositories.

Residual risk: Docker is not a kernel or GPU security boundary. Hostile source
must run only on an approved strongly isolated NVIDIA environment.

### Repudiation

- Candidate attempts, commands, environment resolution, recovery decisions,
  probe decisions, hashes, metrics, confidence, and terminal blockers are
  persisted in run-scoped JSON/JSONL and Markdown artifacts.
- Shared publication remains a human action through a pull request, protected
  branch, or protected release environment.

Residual risk: local artifact owners can edit files after a run. Signed
attestations and centralized retention are deployment responsibilities.

### Information disclosure

- Historical build hooks or benchmarks could read host secrets. Candidate
  subprocesses receive a minimal allowlisted environment; Docker runs do not
  mount host credential stores or inherit host environment variables.
- Logs could contain credentials and then be shared or sent to a model. Common
  credential forms are redacted before model handoff, and every finalized run
  receives a value-free `security_scan.json`.
- Outbound network is an exfiltration channel. Reconstruction needs package
  downloads, so Docker retains bridge-network egress but not host networking.

Residual risk: local modes can read host files, and Docker egress is not
domain-allowlisted. Use `docker-reconstruct` on a credential-free dedicated host
and enforce deployment-level egress allowlists for sensitive investigations.

### Denial of service

- Candidate workloads can consume CPU, memory, disk, processes, GPU memory, and
  network bandwidth. The engine enforces timeouts and bounded retries; Docker
  limits capabilities and process count; probe loops are bounded.
- Environment and cache growth can exhaust disk. Preflight checks and optional
  probe-environment cleanup reduce this risk.

Residual risk: GPU hangs, kernel faults, and storage exhaustion can affect the
host. Use disposable infrastructure, quotas, monitoring, and periodic rebuilds.

### Elevation of privilege

- Containers run with `no-new-privileges`, a bounded process count, read-only
  source/tooling mounts, and no Docker socket. Reconstruction drops all Linux
  capabilities, then adds only `CHOWN`, `DAC_OVERRIDE`, `FOWNER`, `FSETID`,
  `SETGID`, `SETUID`, and `SETFCAP` for candidate-native package installation.
- Shell evaluation of serialized runner arguments has been removed; arguments
  cross the container boundary as JSON.
- LLM-generated shell is never executed generally. Only exact read-only
  diagnostics are accepted and are launched without a shell.

Residual risk: the container currently runs as root inside its namespace and has
GPU device access. Its ephemeral root filesystem is writable because historical
installers invoke the system package manager; target and tooling mounts remain
read-only. Approved infrastructure must account for container, driver, and
kernel escape risk.

## Prompt injection and persistent context

Candidate repositories and logs are untrusted data. Prompt text explicitly marks
them as data, but the actual controls are outside the prompt: closed action
vocabularies, exact command allowlists, retry budgets, package allowlists,
read-only mounts, credential isolation, and deterministic verdict ownership.
No cross-run model memory is used. Run artifacts can persist malicious text, so
they must not be fed to a different privileged agent without human review.

## Rule-of-Three assessment

The optional LLM path combines nondeterministic behavior with external candidate
text. It must therefore not receive NVIDIA secrets or broad internal data. The
candidate runtime receives neither LLM credentials nor host credential mounts.
An internal investigation that needs sensitive source must use an approved
enterprise endpoint, approved isolated infrastructure, and deployment-level
network controls; it must not mix that context with unreviewed external source.

## Reassessment triggers

Re-run the formal TAVA and security tests when adding a tool, model provider,
credential, network destination, mount, shared-resource write, persistent memory,
new Skill/MCP, container privilege, or High/Critical finding.
