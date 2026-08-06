<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Contributing

Follow the repository-level `AGENTS.md` and contribution guide. From the Isaac
Lab repository root, install the agent test extra and run:

```bash
python3 -m pip install --editable "./tools/perf_bisection[test]"
ruff check tools/perf_bisection tools/perf_smoke_test
ruff format --check tools/perf_bisection tools/perf_smoke_test
pytest -q tools/perf_bisection/tests
python3 tools/skills/cli.py check
```

Keep pull requests focused. Bug fixes must include a regression test that fails
without the fix and passes with it.

Public API changes must be additive or go through a deprecation period. Preserve
the JSON artifact contracts and structured failure categories unless the change
includes explicit migration guidance.

Do not add LLM behavior to deterministic measurement, threshold, or search-path
decisions. LLM policies must remain optional, bounded, and auditable.

GPU or environment-compatibility changes should include the corresponding
`probe_range.json`, host configuration, and tooling SHA in the pull-request
description.
