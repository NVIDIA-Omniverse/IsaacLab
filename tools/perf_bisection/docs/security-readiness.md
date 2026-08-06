<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# NVIDIA Agent Security Readiness

Audit baseline: NVIDIA Agent Security Readiness portal, updated August 3, 2026.

Current status: **conditionally ready for public review, not approved for NVIDIA
internal deployment**. Repository-enforceable controls are present, but the
owner must complete the NVIDIA registration, governance, infrastructure, and
system-testing gates below. Do not attest compliance until every required gate
has objective evidence.

## Repository controls

Implemented:

- Deterministic metric classification is isolated from optional LLM policies.
- LLM modes are off by default, require an explicit secure endpoint, redact
  common credential forms, use bounded action vocabularies, and cannot execute
  arbitrary model-generated shell.
- Candidate processes receive a minimal environment with credentials removed.
- Docker reconstruction uses read-only source/tooling mounts, no host network,
  no Docker socket, `no-new-privileges`, and a process limit. It drops all
  capabilities before adding only the seven file-ownership and identity
  capabilities required by candidate-native package managers. Base images are
  pinned by immutable manifest digest.
- Serialized runner arguments cross the container boundary as JSON, not `eval`.
- Real runs require an explicit human target-code trust confirmation; plan
  identifiers, tooling paths, and writable run paths reject traversal, while
  legacy extra arguments cannot override protected runner settings.
- Timed-out commands terminate their complete process session rather than only
  the immediate child.
- Every finalized run receives a value-free credential scan report.
- Security CI audits Python dependencies and the built container; GitHub secret
  scanning and push protection are enabled.
- Upstream Skills are optional, NVIDIA-owned, commit-pinned, and outside the
  verdict path. The utility prints installation commands but does not install
  them.
- Release publication names a GitHub `release` environment so administrators can
  require a final human approval.
- Threat model, security test plan, credential handling, incident response, and
  deployment restrictions are documented.

Residual limitations:

- Docker with GPU access is not a hostile kernel/driver sandbox.
- Reconstruction needs outbound package-network access; repository code cannot
  enforce a deployment-specific domain allowlist.
- Local runner modes execute trusted candidate code with the user's filesystem
  identity.
- The reconstruction container runs as root with a writable ephemeral root
  filesystem because candidate-native installers use the system package
  manager. Target and tooling mounts remain read-only.
- Public GitHub scanning is not a substitute for NVIDIA nSpect malware,
  vulnerability, and secret evidence.

## Mandatory NVIDIA owner gates

The following remain open until completed in NVIDIA systems:

1. Register the agent, each shipped Skill, source repository, Python package,
   container image, and release artifacts in nSpect with the correct program
   classes.
2. Run and pass nSpect vulnerability, secret, and malware scans. Resolve every
   Critical and High finding.
3. Initiate and complete PLC, including security tasks and release gate.
4. Convert `security-threat-model.md` into the authoritative repository-scoped
   SecNemo/Security Architecture STRIDE TAVA and obtain required review.
5. Create agent and Skill cards using the current PLC task templates. Repository
   drafts cannot define the authoritative format.
6. Review the portal and attest only after underlying controls and evidence are
   complete.
7. Attach the architecture/design evidence and `security-test-plan.md` to PLC.
8. Select and document an approved infrastructure pattern. Never run autorun on
   an end-user device or SSH from an autorun agent to a non-sandboxed host.
9. Execute NVIDIA unit, integration, and system agent-security tests in approved
   strong isolation and retain evidence.
10. Register the native `.agents/skills/` aliases with the NVIDIA Skills
    scanning and catalog pipeline before broad publication.

## Human-controlled repository settings

These settings are sources of truth and must be configured and reviewed by a
human repository administrator:

- Protect `develop` and `main`; disallow direct pushes and force pushes.
- Require at least one non-author review, resolved conversations, and passing
  `CI` and `Security` checks before merge.
- Protect the `release` environment with a required human reviewer.
- Keep secret scanning and push protection enabled; enable validity checks,
  non-provider patterns, and Dependabot security updates where available.
- Restrict tag creation and package publication to release owners.
- Review every dependency, action major-version update, container digest update,
  workflow change, and release.

The agent must not configure, weaken, or bypass these controls.

## Guidelines applicability

- Coding agents/tools: applicable. Operators must use approved enterprise tools
  and approved autorun deployment patterns.
- Approved LLM endpoints and allowable data: applicable only when an LLM mode is
  enabled. Public, Confidential, and Secret data require an approved endpoint;
  Regulated data requires case-by-case Legal approval.
- SSH: the agent has no SSH feature. Operators must not add an unsandboxed SSH
  escape path around the documented runtime boundary.
- External plugins: applicable to Skills and future MCP/hooks. Only the pinned
  NVIDIA upstream handoffs are listed. Non-NVIDIA plugins must run in isolation;
  self-built Skills require NV-CARPS onboarding for wide internal use.
- Shared resources: applicable to source control and release artifacts. The
  software creates local drafts only; final pushes, merges, tags, publications,
  and governance changes remain human actions.
- Least privilege and credentials: applicable. Do not pass full environments or
  credential mounts; use scoped, auditable, revocable credentials with short
  lifetimes.
- Output hygiene and incidents: applicable. Scan before sharing; on an incident,
  rotate credentials, preserve evidence, report to CSIRT, and wait before
  teardown.
- Slackbot, browser/computer use, MCP, and production service write guidance: not
  currently applicable because this repository provides none of those
  capabilities. Reassess before adding one.

## Evidence maintenance

Re-run this audit, TAVA, scans, and security testing whenever tools, Skills,
credentials, LLM endpoints, data classes, mounts, network access, container
privileges, shared-resource writes, infrastructure, or release artifacts change.
