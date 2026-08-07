<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Optional LLM policies

LLM recovery and container probing are disabled unless explicitly selected.
They use an OpenAI-compatible chat endpoint and never participate in metric
classification or binary-search decisions.

No provider is selected by default. Always pass `--base_url`; remote endpoints
must use HTTPS and URLs containing credentials are rejected. NVIDIA users must
select an endpoint approved by the Agent Security Readiness portal (normally
`inference-api.nvidia.com` when authorized for NVIDIA data or internal network
access, or `build.nvidia.com` only in the documented isolated use case). Do not
use a personal provider key with NVIDIA data.

Deployments can enforce an exact hostname allowlist with the comma-separated
`ISAACLAB_BISECTION_LLM_HOSTS` environment variable. Redirects are rejected so
the bearer token and evidence cannot move to a second endpoint. Non-global IP
literals are rejected except explicit loopback development URLs.

Inject the scoped, short-lived credential in an environment variable:

```bash
export OPENAI_API_KEY=<secret>
```

Verify recovery connectivity before a GPU run:

```bash
isaaclab-bisect recovery-selftest \
    --recovery llm \
    --model <MODEL> \
    --base_url <OPENAI_COMPATIBLE_BASE_URL>
```

Verify the bounded setup probe:

```bash
isaaclab-bisect probe-selftest \
    --model <MODEL> \
    --base_url <OPENAI_COMPATIBLE_BASE_URL>
```

Use `--api_key_env` when credentials are stored under another environment
variable. Do not put keys in plans, command lines, URLs, logs, or artifacts.
NVIDIA bearer tokens exposed to the process must have a maximum lifetime of one
hour and should be rotated after the session. Regulated data must not be sent to
an LLM without case-by-case Legal approval.

Recovery is limited to the action vocabulary and retry budget enforced in
Python. Invalid responses fall back to deterministic behavior. Probe actions
are similarly bounded and cannot emit `GOOD`, `BAD`, or `UNCLEAR`. Candidate
logs are untrusted and may contain prompt injection. The probe can run only the
exact read-only diagnostics `docker info`, `docker version`, `nvidia-smi`, and
`df -h`; validation happens outside the model and commands launch without a
shell. Common credential forms are redacted before model handoff.
