#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate a dev perf map JSON fixture from real IsaacLab commit SHAs.

The perf map assigns simulated fps_mean values to real commit SHAs, creating
an artificial performance regression at a known commit for end-to-end testing.

Usage::

    python tests/build_dev_perf_map.py \\
        --repo ../IsaacLab \\
        --good-count 4 \\
        --bad-count 4 \\
        --good-fps 3800.0 \\
        --bad-fps 2400.0 \\
        --output tests/dev_perf_map.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

_SHA_SHORT = 12  # chars used for display


def _get_commits(repo: str, n: int) -> list[str]:
    """Return the n most recent non-merge commit SHAs, newest-first."""
    result = subprocess.run(
        ["git", "-C", repo, "log", "--format=%H", "--no-merges", f"-{n}"],
        capture_output=True,
        text=True,
        check=True,
    )
    shas = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    if len(shas) < n:
        print(
            f"Warning: requested {n} commits but only found {len(shas)} in repo",
            file=sys.stderr,
        )
    return shas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a dev perf map from real IsaacLab commit SHAs"
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Path to the IsaacLab git repository",
    )
    parser.add_argument(
        "--good-count",
        type=int,
        default=4,
        help="Number of good (high-fps) commits (default: 4)",
    )
    parser.add_argument(
        "--bad-count",
        type=int,
        default=4,
        help="Number of bad (low-fps) commits (default: 4)",
    )
    parser.add_argument(
        "--good-fps",
        type=float,
        default=3800.0,
        help="Simulated fps_mean for good commits (default: 3800.0)",
    )
    parser.add_argument(
        "--bad-fps",
        type=float,
        default=2400.0,
        help="Simulated fps_mean for bad commits (default: 2400.0)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the output JSON perf map",
    )
    args = parser.parse_args()

    total = args.good_count + args.bad_count

    # git log returns newest-first; fetch enough commits then reverse to oldest-first
    shas_newest_first = _get_commits(args.repo, total)
    if len(shas_newest_first) < total:
        print(
            f"Error: need {total} commits but repo only has {len(shas_newest_first)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Reverse so index 0 = oldest, index (total-1) = newest
    shas_oldest_first = list(reversed(shas_newest_first))

    # Assign fps: first good_count = good_fps, last bad_count = bad_fps
    perf_map: dict[str, float] = {}
    for i, sha in enumerate(shas_oldest_first):
        if i < args.good_count:
            perf_map[sha] = args.good_fps
        else:
            perf_map[sha] = args.bad_fps

    # Identify key SHAs
    good_shas = shas_oldest_first[: args.good_count]
    bad_shas = shas_oldest_first[args.good_count :]

    last_good_sha = good_shas[-1]   # last of the good range (used as --good arg)
    first_bad_sha = bad_shas[0]     # first of the bad range (expected bisect result)
    last_bad_sha = bad_shas[-1]     # last of the bad range (used as --bad arg)

    # Add special keys for test script convenience
    perf_map["good_sha"] = last_good_sha       # type: ignore[assignment]
    perf_map["bad_sha"] = last_bad_sha         # type: ignore[assignment]
    perf_map["expected_first_bad"] = first_bad_sha  # type: ignore[assignment]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(perf_map, f, indent=2)
        f.write("\n")

    # Print summary
    print(f"Wrote perf map to {output_path}")
    print(f"Total commits: {total} ({args.good_count} good + {args.bad_count} bad)")
    print(f"Oldest commit (index 0): {shas_oldest_first[0][:_SHA_SHORT]}")
    print(f"Newest commit (index {total-1}): {shas_oldest_first[-1][:_SHA_SHORT]}")
    print()
    print("Good range (fps={:.0f}):".format(args.good_fps))
    for sha in good_shas:
        print(f"  {sha[:_SHA_SHORT]}")
    print()
    print("Bad range (fps={:.0f}):".format(args.bad_fps))
    for sha in bad_shas:
        marker = " <-- expected_first_bad" if sha == first_bad_sha else ""
        print(f"  {sha[:_SHA_SHORT]}{marker}")
    print()
    print("Key SHAs for bisect command:")
    print(f"  --good {last_good_sha[:_SHA_SHORT]}  (last good)")
    print(f"  --bad  {last_bad_sha[:_SHA_SHORT]}  (last bad)")
    print(f"  expected first_bad_sha: {first_bad_sha[:_SHA_SHORT]}")


if __name__ == "__main__":
    main()
