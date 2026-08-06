# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recovery policies for the bisection agent's benchmark-execution friction.

Running an IsaacLab benchmark on a freshly checked-out commit rarely succeeds on
the first try: stale Kit/JIT caches, transient downloads, headless/EULA quirks,
and slow first-run kernel compilation routinely make a *runnable* commit look
broken. If every such hiccup were recorded as a skip (or, worse, a regression),
the bisection would riddle its range with holes or condemn innocent commits.

This module owns the *recovery* decision -- "given a failed measurement, should
we inspect, retry (and how), or accept that this commit is genuinely
un-evaluable?" -- and deliberately keeps it separate from the *verdict*. A
recovery policy never decides GOOD/BAD/UNCLEAR; it only decides how hard to try
before the deterministic comparator gets a clean measurement (or a clean skip).

Two policies are provided:

* :class:`DeterministicRecoveryPolicy` encodes the ``failure -> inspect -> retry
  -> accept`` loop with a bounded budget and is the default (works with
  ``--recovery deterministic`` and needs no model/API key).
* :class:`NoRecoveryPolicy` accepts the first outcome as-is (``--recovery none``).

An optional LLM-backed policy lives in :mod:`bisection.recovery_llm` and
implements the same :class:`RecoveryPolicy` interface, so the engine is agnostic
to which one drives recovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

# Recovery actions the engine knows how to apply. ``accept`` ends the loop; the
# ``retry_*`` actions map to concrete runner knobs via :func:`knobs_for_action`.
ACTION_ACCEPT = "accept"
ACTION_RETRY_PLAIN = "retry_plain"
ACTION_RETRY_CLEAR_CACHES = "retry_clear_caches"
ACTION_RETRY_INCREASE_TIMEOUT = "retry_increase_timeout"
ACTION_RETRY_REINSTALL = "retry_reinstall"

RETRY_ACTIONS = frozenset(
    {ACTION_RETRY_PLAIN, ACTION_RETRY_CLEAR_CACHES, ACTION_RETRY_INCREASE_TIMEOUT, ACTION_RETRY_REINSTALL}
)

# Default timeout extension (seconds) applied by a ``retry_increase_timeout`` action.
DEFAULT_TIMEOUT_EXTENSION_S = 600

# Log substrings that indicate a runtime/ABI/import failure (the commit's env
# cannot run on this machine) rather than transient infra friction.
_RUNTIME_INCOMPAT_SIGNATURES = (
    "ModuleNotFoundError",
    "ImportError:",
    "undefined symbol",
    "GLIBC_",
    "cannot open shared object file",
)

# Host/operator-level blockers: the commit's environment was never the problem, so
# retrying the same measurement on the same machine cannot help. Each entry maps a
# specific skip category to the log substrings (matched case-insensitively) that
# identify it. These are checked before commit-level classification so a useful,
# self-hosted agent reports "your Docker daemon is down" instead of a generic
# ``infra`` skip and does not waste its retry budget on an unfixable condition.
_HOST_BLOCKER_SIGNATURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "host_resource",
        ("no space left on device",),
    ),
    (
        "docker_unavailable",
        (
            "cannot connect to the docker daemon",
            "is the docker daemon running",
            "docker: command not found",
            "permission denied while trying to connect to the docker daemon socket",
        ),
    ),
    (
        "gpu_unavailable",
        (
            "could not select device driver",
            "nvidia-container-cli",
            "failed to initialize nvml",
            "no cuda-capable device is detected",
            "cuda driver version is insufficient",
        ),
    ),
    (
        "base_image_missing",
        (
            "unable to find image",
            "manifest unknown",
            "pull access denied for",
            "repository does not exist or may require",
        ),
    ),
)

# Source-checkout failures are infra-level (the commit content is fine, the clone
# is not) but, unlike host blockers, a retry into a fresh source dir can recover
# them, so they stay on the normal retry path with a more specific label than
# ``infra``.
_SOURCE_CHECKOUT_SIGNATURES = (
    "did not match any file(s) known to git",
    "reference is not a tree",
    "fatal: not a git repository",
    "could not read from remote repository",
    "unable to create '.git/index.lock'",
    "unable to create '/candidate/.git/index.lock'",
)


def classify_host_blocker(log_tail: str) -> str | None:
    """Return a non-retryable host/operator skip category for ``log_tail``, or None.

    These blockers (disk full, Docker daemon down, GPU driver/toolkit missing,
    base image absent) live on the machine running the bisection, not in the
    commit under test, so the agent should stop rather than retry.
    """
    lowered = log_tail.lower()
    for category, signatures in _HOST_BLOCKER_SIGNATURES:
        if any(signature in lowered for signature in signatures):
            return category
    return None


def looks_source_checkout_failure(log_tail: str) -> bool:
    """Return True if ``log_tail`` shows the source clone/checkout (not the env) failed."""
    lowered = log_tail.lower()
    return any(signature in lowered for signature in _SOURCE_CHECKOUT_SIGNATURES)


@dataclass
class RecoveryKnobs:
    """Concrete execution overrides applied to the next runner invocation."""

    clear_caches: bool = False
    force_reinstall: bool = False
    extra_timeout_s: int | None = None

    def runner_args(self) -> list[str]:
        """Return the extra CLI flags this set of knobs adds to the runner command."""
        args: list[str] = []
        if self.clear_caches:
            args.append("--clear_caches")
        if self.force_reinstall:
            args.append("--force_reinstall")
        return args


@dataclass(frozen=True)
class RecoveryContext:
    """Everything a policy needs to decide what to do after a failed measurement."""

    commit_sha: str
    label: str
    run_idx: int
    attempt: int
    note: str | None
    exit_code: int | None
    timed_out: bool
    artifact_dir: Path
    log_tail: str = ""
    env_status: tuple[str | None, str | None, str | None] | None = None


@dataclass(frozen=True)
class RecoveryDecision:
    """A policy's decision: retry (with an action) or accept the outcome as a skip."""

    action: str
    reason: str
    skip_category: str | None = None

    def to_json(self) -> dict:
        """Serialize the decision for the audit log."""
        return {"action": self.action, "reason": self.reason, "skip_category": self.skip_category}


class RecoveryPolicy(Protocol):
    """Decides how to recover from a measurement that produced no usable metric."""

    def decide(self, ctx: RecoveryContext) -> RecoveryDecision:
        """Return a :class:`RecoveryDecision` for the failed measurement ``ctx``."""
        ...


def knobs_for_action(action: str, previous: RecoveryKnobs) -> RecoveryKnobs:
    """Translate a retry action into concrete runner knobs, accumulating prior ones.

    Knobs accumulate so escalation is monotonic (e.g. once caches are cleared they
    stay cleared on subsequent retries).
    """
    knobs = RecoveryKnobs(
        clear_caches=previous.clear_caches,
        force_reinstall=previous.force_reinstall,
    )
    if action == ACTION_RETRY_CLEAR_CACHES:
        knobs.clear_caches = True
    elif action == ACTION_RETRY_INCREASE_TIMEOUT:
        knobs.extra_timeout_s = DEFAULT_TIMEOUT_EXTENSION_S
    elif action == ACTION_RETRY_REINSTALL:
        knobs.force_reinstall = True
        knobs.clear_caches = True
    return knobs


def looks_runtime_incompatible(log_tail: str) -> bool:
    """Return True if a log excerpt shows an environment/ABI/import failure signature."""
    return any(signature in log_tail for signature in _RUNTIME_INCOMPAT_SIGNATURES)


def _skip_category_for(note: str | None, log_tail: str) -> str:
    """Map a friction note to a clean skip category for reporting."""
    # Host/operator blockers take precedence over any note: they point at the
    # machine running the bisection, not the commit, and are the most actionable
    # thing to tell the user.
    host = classify_host_blocker(log_tail)
    if host is not None:
        return host
    if note and note.startswith("env_skip:"):
        return note.split(":", 1)[1] or "runtime_incompatible"
    if note == "probe_failed:plan_issue":
        return "plan_issue"
    if note == "probe_failed:harness_blocked":
        return "harness_blocked"
    if note == "candidate_timeout":
        return "runtime_incompatible"
    if looks_runtime_incompatible(log_tail):
        return "runtime_incompatible"
    if looks_source_checkout_failure(log_tail):
        return "source_checkout_failed"
    return "infra"


@dataclass
class NoRecoveryPolicy:
    """Accept the first failed outcome as a skip; never retry (``--recovery none``)."""

    def decide(self, ctx: RecoveryContext) -> RecoveryDecision:
        """Immediately accept the outcome, classifying its skip category."""
        return RecoveryDecision(
            ACTION_ACCEPT,
            "recovery disabled; accepting first outcome",
            _skip_category_for(ctx.note, ctx.log_tail),
        )


@dataclass
class DeterministicRecoveryPolicy:
    """Bounded ``failure -> inspect -> retry -> accept`` recovery with no model.

    The policy inspects the friction note and log tail and picks a targeted retry
    (clear caches, extend timeout, reinstall, or a plain retry). A pinned
    dependency that is simply unavailable is never retried. Once the retry budget
    is exhausted, the commit is accepted as an honest skip with a clean category.
    """

    max_attempts: int = 2

    def decide(self, ctx: RecoveryContext) -> RecoveryDecision:
        """Return the next recovery step for the failed measurement ``ctx``."""
        note = ctx.note or ""

        # Host/operator blockers (disk full, Docker daemon down, GPU driver/toolkit
        # missing, base image absent) live on this machine, not in the commit, so a
        # retry cannot help. Accept immediately with the specific category instead of
        # spending the retry budget on an unfixable condition.
        env_detail = ctx.env_status[2] if ctx.env_status is not None and ctx.env_status[2] else ""
        host = classify_host_blocker("\n".join(part for part in (ctx.log_tail, env_detail) if part))
        if host is not None:
            return RecoveryDecision(
                ACTION_ACCEPT, f"host/operator blocker ({host}); not fixable by retrying on this machine", host
            )

        # A pin that does not exist on the index cannot be fixed by retrying.
        if note.startswith("env_skip:dependency_unavailable"):
            return RecoveryDecision(
                ACTION_ACCEPT, "pinned dependency unavailable on index; not retryable", "dependency_unavailable"
            )

        if note.startswith("env_skip:perf_smoke_tooling_incompatible"):
            return RecoveryDecision(
                ACTION_ACCEPT,
                "candidate APIs do not satisfy the pinned perf-smoke tooling contract; not retryable",
                "perf_smoke_tooling_incompatible",
            )

        # The runner already resets and re-clones once internally before emitting this
        # skip, so a terminal source-checkout failure means the commit is unreachable
        # from the local repo (a fresh runner would repeat the same failing fetch).
        if note.startswith("env_skip:source_checkout_failed"):
            return RecoveryDecision(
                ACTION_ACCEPT,
                "source checkout failed after an in-runner re-clone; not retryable",
                "source_checkout_failed",
            )

        # The probe has already inspected setup state. Respect terminal probe
        # outcomes instead of wrapping them in generic retries.
        if note.startswith("probe_failed:"):
            return RecoveryDecision(
                ACTION_ACCEPT,
                "probe phase stopped before deterministic benchmarking",
                _skip_category_for(note, ctx.log_tail),
            )

        if ctx.attempt >= self.max_attempts:
            return RecoveryDecision(
                ACTION_ACCEPT,
                f"recovery budget exhausted after {ctx.attempt} retr{'y' if ctx.attempt == 1 else 'ies'}",
                _skip_category_for(note, ctx.log_tail),
            )

        # A failed install may be a transient download; try one clean reinstall.
        if note.startswith("env_skip:install_failed"):
            return RecoveryDecision(
                ACTION_RETRY_REINSTALL, "install failed; retrying a clean reinstall (possible transient download)"
            )

        # Runtime/import/ABI failures: a stale Kit/JIT cache is the usual culprit;
        # clear it once, then accept as runtime-incompatible if it persists.
        if note.startswith("env_skip:runtime_incompatible") or looks_runtime_incompatible(ctx.log_tail):
            if ctx.attempt == 0:
                return RecoveryDecision(
                    ACTION_RETRY_CLEAR_CACHES, "runtime/import failure; clearing stale Kit/JIT caches and retrying"
                )
            return RecoveryDecision(
                ACTION_ACCEPT, "runtime/import failure persisted after clearing caches", "runtime_incompatible"
            )

        # A timeout may just be first-run kernel compilation; extend once.
        if note == "candidate_timeout":
            if ctx.attempt == 0:
                return RecoveryDecision(
                    ACTION_RETRY_INCREASE_TIMEOUT, "timed out; extending the timeout once for first-run compilation"
                )
            return RecoveryDecision(ACTION_ACCEPT, "timed out again with an extended budget", "runtime_incompatible")

        # Generic infra failure (nonzero exit, missing result, metric unavailable):
        # a plain retry first, then a cache clear, then accept.
        if ctx.attempt == 0:
            return RecoveryDecision(ACTION_RETRY_PLAIN, "transient runner failure; plain retry")
        return RecoveryDecision(ACTION_RETRY_CLEAR_CACHES, "runner failure persisted; retrying with cleared caches")


@dataclass
class RecoveryEvent:
    """One recorded recovery step, appended to a measurement's attempt record."""

    attempt: int
    note: str | None
    decision: str
    reason: str
    skip_category: str | None = None
    knobs: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        """Serialize the recovery event for artifacts/audit."""
        return {
            "attempt": self.attempt,
            "note": self.note,
            "decision": self.decision,
            "reason": self.reason,
            "skip_category": self.skip_category,
            "knobs": self.knobs,
        }


def build_policy(kind: str, **kwargs) -> RecoveryPolicy:
    """Construct a recovery policy by name (``none`` or ``deterministic``).

    The ``llm`` policy is built by :mod:`bisection.recovery_llm` because it needs
    a model client; this factory covers the model-free policies.
    """
    if kind == "none":
        return NoRecoveryPolicy()
    if kind == "deterministic":
        return DeterministicRecoveryPolicy(**kwargs)
    raise ValueError(f"unknown model-free recovery policy: {kind!r}")
