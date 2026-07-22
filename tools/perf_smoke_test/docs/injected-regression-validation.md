<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Injected Regression Validation Range

This disposable local branch validates the performance bisection agent against
a real IsaacLab benchmark. The next commit deliberately slows the direct
Cartpole task; a later documentation-only commit keeps the branch tip bad so
the bisection search must inspect multiple commits.

The deliberately injected first-bad commit is
`fcc8d462e8d5c4cd61b63d26afd7e202f8c1e089`.
