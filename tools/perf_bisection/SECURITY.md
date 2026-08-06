<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Security

Report suspected vulnerabilities privately to the repository maintainers rather
than opening a public issue.

## Trust model

The agent checks out and installs historical source code, invokes package
installers, and runs candidate workloads. Treat every target repository and
commit as trusted code. Use `docker-reconstruct` on a dedicated host for
investigations that require stronger isolation; the container is not a security
sandbox against hostile kernel or GPU workloads.

Local runner modes execute candidate code with the user's identity and are not
appropriate for untrusted repositories. Docker modes mount only the target
repository and run-scoped caches, constrain Linux capabilities, prevent new
privileges, and do not use host networking. Reconstruction retains only the
file-ownership and identity capabilities required by candidate-native package
managers. It still has outbound network access because reconstruction downloads
pinned packages.

Every real runner invocation requires `--trust_target_code`; supply it only
after a human reviews the repository identity and exact candidate/tooling
history as executable code. This confirmation does not make hostile code safe.
Run-scoped writable paths and tooling-relative paths are confined and traversal
is rejected. Legacy runner passthrough accepts only `--install_scope`; security,
repository, tooling, cache, and artifact flags cannot be overridden.

## Credentials and data

Do not place API keys in plans, command-line arguments, URLs, logs, reports, or
artifacts. Candidate subprocesses receive a minimal environment that excludes
LLM, GitHub, SSH-agent, cloud, and unrelated shell credentials. Do not mount
`~/.ssh`, `~/.aws`, credential stores, a home directory, or a Docker socket into
the reconstruction container.

LLM modes are disabled unless explicitly selected and require an explicit HTTPS
endpoint. NVIDIA users must use an approved enterprise endpoint, must not use a
personal API key with NVIDIA data, and must not process Regulated data without
case-by-case Legal approval. Inject only the scoped credential needed for the
request, keep bearer-token lifetime at or below one hour, and rotate it after
the session. Logs and plans sent to the model are untrusted and may contain
prompt injection; model-requested diagnostics are enforced by a fixed,
read-only command allowlist outside the model.

Every completed run writes `security_scan.json`. A `blocked` result means the
artifacts must not be shared until the possible credential is removed and the
credential is rotated. Re-run the scan explicitly with:

```bash
isaaclab-bisect-scan-artifacts <OUTPUT_DIR>
```

## Shared resources and releases

The agent does not merge pull requests, push protected branches, publish
containers, or update issue trackers. Treat generated changes and reports as
drafts. A human must review and approve the final push, merge, tag, package
publication, container publication, or other source-of-truth write. Protect
`main` and configure the `release` GitHub environment with required reviewers;
the publish workflow names that environment but repository administrators must
enforce its protection rules.

Review generated relaunch commands before running them on another host. Runner
extra arguments and custom command templates are operator-controlled code
execution surfaces.

## NVIDIA deployment gates

Public source availability is not NVIDIA deployment approval. Before internal
deployment, the owner must complete the open gates recorded in
`docs/security-readiness.md`: nSpect registration and scans, PLC, a
repository-scoped TAVA, agent and Skill cards in the authoritative PLC format,
portal attestation, approved infrastructure selection, and isolated unit,
integration, and system security testing. Self-built Skills must be provisioned
through NV-CARPS before broad internal sharing.

## Security incidents at NVIDIA

If prompt injection, jailbreak, unexpected dangerous behavior, credential
exposure, or compromise is observed, immediately revoke and rotate credentials.
Preserve the environment, logs, container state, repository, and shell history;
do not terminate or rebuild an ephemeral environment. Report to
`csirt@nvidia.com` (`csirt911@nvidia.com` for an active critical incident), then
wait for CSIRT direction before cleanup.
