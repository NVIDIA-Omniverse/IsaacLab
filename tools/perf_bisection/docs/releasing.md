<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Publishing

Isaac Lab is the source of truth for the bisection runtime and its Agent Skills.
The package remains in the `0.x` version series while its plugin and artifact
contracts stabilize. Do not publish a separate source repository.

## Release checklist

1. Run linting and the complete CPU-only test suite.
2. Build the wheel and install it into a fresh virtual environment.
3. Build the Docker image for both `linux/amd64` and `linux/arm64`.
4. Run a synthetic bisection from the installed console command.
5. Run a short `docker-reconstruct` range on a supported GPU host.
6. Verify `report.md`, `summary.json`, `probe_range.json`, and attempt artifacts.
7. Confirm `security_scan.json` passes and Security CI reports no unresolved
   High/Critical dependency or container vulnerability.
8. For NVIDIA deployment, attach current nSpect vulnerability, secret, and
   malware evidence; confirm PLC/TAVA/cards/testing/portal gates are complete.
9. Confirm all documented compatibility and security limits remain accurate.
10. Obtain human approval before merging the exact tested commit into the
    Isaac Lab development branch.

After merge, coordinate with the NVIDIA Skills publishing owners to scan and
mirror the aliases under `.agents/skills/` into `nvidia/skills`. The native
Isaac Lab skill remains authoritative; the catalog entry is a distribution
mirror.

The current distribution does not publish a container image. Users build the
pinned Dockerfile from their Isaac Lab checkout. Any future package or container
publication requires a separately reviewed workflow and immutable artifact
provenance.

## Clean-clone acceptance

Before announcing a release to other developers, validate from a new clone on a
host that does not have the source checkout on `PYTHONPATH`:

```bash
python3 -m venv /tmp/isaaclab-bisect-release
source /tmp/isaaclab-bisect-release/bin/activate
python -m pip install ./tools/perf_bisection
isaaclab-bisect --help
```

Then run one synthetic workflow and one Docker reconstruction smoke test against
an explicitly supplied `--repo_root`.
