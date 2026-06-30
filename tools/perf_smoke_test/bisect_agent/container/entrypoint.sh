#!/usr/bin/env bash
# entrypoint.sh — bisect runner entrypoint for each commit under test.
#
# Required environment variables:
#   COMMIT_SHA   — the commit to check out and benchmark
#   TASK_ID      — IsaacLab task identifier, e.g. Isaac-Velocity-Flat-G1-Direct
#   BACKEND      — backend key, e.g. newton
#
# Output:
#   /artifacts/perf_smoke_test_result.json  — written by Phase 2 (build_bench_result.py)
#
# TODO: validate this script against a real nvcr.io/nvidia/isaac-sim:4.5.0 image before
# promoting to production. Specifically verify:
#   - the mount point for the IsaacLab repo (/isaaclab)
#   - that ./isaaclab.sh -i none --quiet succeeds in the bundled Python environment
#   - that benchmark_non_rl.py accepts --benchmark_backend json

set -e

: "${COMMIT_SHA:?COMMIT_SHA environment variable is required}"
: "${TASK_ID:?TASK_ID environment variable is required}"
: "${BACKEND:?BACKEND environment variable is required}"

ARTIFACTS_DIR="/artifacts"
mkdir -p "${ARTIFACTS_DIR}"

LOG_FILE="${ARTIFACTS_DIR}/benchmark.log"

cd /isaaclab

echo "[entrypoint] Fetching and checking out ${COMMIT_SHA}" | tee -a "${LOG_FILE}"
git fetch --all --quiet
git checkout "${COMMIT_SHA}" --quiet

echo "[entrypoint] Installing IsaacLab via ./isaaclab.sh -i none" | tee -a "${LOG_FILE}"
# Must use ./isaaclab.sh -i, NOT bare pip install.
# Isaac Sim ships a bundled Python environment with pre-installed warp/physx/torch.
# Bare pip install risks silently downgrading or corrupting bundled packages.
./isaaclab.sh -i none --quiet 2>&1 | tee -a "${LOG_FILE}"

echo "[entrypoint] Phase 1: running benchmark_non_rl.py for task=${TASK_ID} backend=${BACKEND}" | tee -a "${LOG_FILE}"
BENCH_START=$(date +%s%N)

./isaaclab.sh -p scripts/benchmarks/benchmark_non_rl.py \
    --task "${TASK_ID}" \
    --num_envs 512 \
    --num_frames 300 \
    --benchmark_backend json \
    --output_path "${ARTIFACTS_DIR}" \
    2>&1 | tee -a "${LOG_FILE}"

BENCH_EXIT=${PIPESTATUS[0]}
BENCH_END=$(date +%s%N)
WALL_TIME_S=$(echo "scale=3; (${BENCH_END} - ${BENCH_START}) / 1000000000" | bc)

echo "[entrypoint] Phase 1 exit_code=${BENCH_EXIT} wall_time_s=${WALL_TIME_S}" | tee -a "${LOG_FILE}"

echo "[entrypoint] Phase 2: running build_bench_result.py" | tee -a "${LOG_FILE}"
./isaaclab.sh -p tools/perf_smoke_test/build_bench_result.py \
    --task_id "${TASK_ID}" \
    --artifact_dir "${ARTIFACTS_DIR}" \
    --exit_code "${BENCH_EXIT}" \
    --wall_time_s "${WALL_TIME_S}" \
    --timeout_s 600 \
    --log_file "${LOG_FILE}" \
    2>&1 | tee -a "${LOG_FILE}"

echo "[entrypoint] Done. Results in ${ARTIFACTS_DIR}/perf_smoke_test_result.json"
