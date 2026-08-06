# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Security boundaries shared by the bisection runner and optional LLM policies."""

from __future__ import annotations

import ipaddress
import os
import re
import shlex
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

_CANDIDATE_ENV_NAMES = frozenset(
    {
        "ALL_PROXY",
        "CUDA_HOME",
        "CUDA_VISIBLE_DEVICES",
        "CURL_CA_BUNDLE",
        "DISPLAY",
        "HOME",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "LOGNAME",
        "NO_PROXY",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
        "PATH",
        "PERF_BISECT_PROGRESS",
        "REQUESTS_CA_BUNDLE",
        "SHELL",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TEMP",
        "TERM",
        "TMP",
        "TMPDIR",
        "USER",
        "WAYLAND_DISPLAY",
        "XDG_CACHE_HOME",
        "XDG_RUNTIME_DIR",
    }
)
_CANDIDATE_ENV_PREFIXES = ("LC_",)
_PROXY_NAMES = frozenset({"ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY"})

_ALLOWED_PROBE_COMMANDS = frozenset(
    {
        ("df", "-h"),
        ("docker", "info"),
        ("docker", "version"),
        ("nvidia-smi",),
    }
)
_SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
_ALLOWED_RUNNER_EXTRA_FLAGS = frozenset({"--install_scope"})

_REDACTIONS = (
    (re.compile(r"(?i)\b(authorization\s*:\s*bearer\s+)\S+"), r"\1<redacted>"),
    (re.compile(r"(?i)\b(api[_-]?key|token|password|passwd|secret)\s*[=:]\s*([^\s,;]+)"), r"\1=<redacted>"),
    (
        re.compile(r"(?i)\b(https?://)([^/\s:@]+):([^/\s@]+)@"),
        r"\1<redacted>:<redacted>@",
    ),
    (
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?-----END .*?PRIVATE KEY-----", re.DOTALL),
        "<redacted-private-key>",
    ),
)


def candidate_subprocess_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a minimal host environment safe to expose to candidate code.

    Historical commits execute arbitrary package build hooks and benchmark code.
    They must not inherit the agent's LLM key, GitHub token, SSH agent, cloud
    credentials, or unrelated shell configuration.
    """
    source = os.environ if source is None else source
    result = {
        name: value
        for name, value in source.items()
        if name in _CANDIDATE_ENV_NAMES or name.startswith(_CANDIDATE_ENV_PREFIXES)
    }
    for name in _PROXY_NAMES:
        value = result.get(name)
        if value and _url_contains_credentials(value):
            result.pop(name)
    return result


def validate_llm_base_url(base_url: str | None) -> str:
    """Validate and normalize an explicitly selected LLM endpoint."""
    if not base_url:
        raise ValueError(
            "LLM mode requires an explicit --base_url; no public provider is selected by default. "
            "NVIDIA users must choose an approved enterprise endpoint."
        )
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("LLM base URL must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("LLM base URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("LLM base URL must not contain a query string or fragment")
    if parsed.scheme != "https" and parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("remote LLM endpoints must use HTTPS")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_loopback and not address.is_global:
        raise ValueError("LLM base URL must not use a private, link-local, or reserved IP address")
    allowed_hosts = {
        host.strip().lower() for host in os.environ.get("ISAACLAB_BISECTION_LLM_HOSTS", "").split(",") if host.strip()
    }
    if allowed_hosts and parsed.hostname.lower() not in allowed_hosts:
        raise ValueError("LLM endpoint host is not in ISAACLAB_BISECTION_LLM_HOSTS")
    return base_url.rstrip("/")


def validate_path_component(value: str, field: str) -> str:
    """Reject path separators and traversal in identifiers used as directories."""
    if not _SAFE_PATH_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{field} must be one safe path component")
    return value


def validate_relative_path(value: str, field: str) -> Path:
    """Return a normalized relative path that cannot escape its future root."""
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a non-traversing relative path")
    return path


def resolve_path_within(root: Path, value: str | Path, field: str, *, allow_root: bool = False) -> Path:
    """Resolve ``value`` and require it to remain below ``root``."""
    root = root.resolve()
    path = Path(value)
    unresolved = path if path.is_absolute() else root / path
    if unresolved.is_symlink():
        raise ValueError(f"{field} must not be a symbolic link")
    resolved = unresolved.resolve()
    if resolved == root:
        if allow_root:
            return resolved
        raise ValueError(f"{field} must not resolve to the run root")
    if root not in resolved.parents:
        raise ValueError(f"{field} must resolve below the run root: {root}")
    return resolved


def validate_runner_extra_args(arguments: list[str]) -> None:
    """Allow only explicitly modeled, non-security-sensitive legacy runner flags."""
    index = 0
    while index < len(arguments):
        flag = arguments[index].split("=", 1)[0]
        if flag not in _ALLOWED_RUNNER_EXTRA_FLAGS:
            raise ValueError(f"runner.extra_args flag is not allowlisted: {flag}")
        if "=" not in arguments[index]:
            index += 1
            if index >= len(arguments) or arguments[index].startswith("-"):
                raise ValueError(f"runner.extra_args flag requires a value: {flag}")
        index += 1


def parse_probe_debug_command(command: str) -> list[str]:
    """Parse an LLM-requested diagnostic command through a strict allowlist."""
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"invalid diagnostic command syntax: {exc}") from exc
    if tuple(argv) not in _ALLOWED_PROBE_COMMANDS:
        allowed = ", ".join(shlex.join(command) for command in sorted(_ALLOWED_PROBE_COMMANDS))
        raise ValueError(f"diagnostic command is not allowlisted; choose one of: {allowed}")
    return argv


def redact_sensitive_text(text: str) -> str:
    """Redact common credential forms before text leaves the local process."""
    redacted = text
    for pattern, replacement in _REDACTIONS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _url_contains_credentials(value: str) -> bool:
    """Return whether a URL-shaped environment value embeds user information."""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return True
    return parsed.username is not None or parsed.password is not None
