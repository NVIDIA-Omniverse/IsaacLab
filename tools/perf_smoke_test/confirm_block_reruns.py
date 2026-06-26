# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rerun BLOCK cells and annotate artifacts with confirmation FPS attempts."""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdicts_file", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--ci_image_tag", required=True)
    parser.add_argument("--reruns", type=int, default=2)
    return parser.parse_args()


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value)


def _extract_fps_series(perf_info: list[dict]) -> list[float]:
    for phase in perf_info:
        if phase.get("phase_name") == "runtime":
            for measurement in phase.get("measurements", []):
                value = measurement.get("value", {})
                if measurement.get("name", "").endswith("Step Frametimes") and isinstance(value, dict):
                    return list(value.get("Environment step effective FPS", []))
    return []


def _excluded_frames(launch_config: dict) -> frozenset[int]:
    indices: set[int] = set()
    for entry in launch_config.get("excluded_frames_raw") or []:
        if isinstance(entry, list):
            indices.update(range(int(entry[0]), int(entry[1]) + 1))
        else:
            indices.add(int(entry))
    return frozenset(indices)


def _gate_mean_fps(perf_info_path: Path, launch_config: dict) -> float:
    with perf_info_path.open() as fh:
        series = _extract_fps_series(json.load(fh))
    excluded = _excluded_frames(launch_config)
    filtered = [fps for idx, fps in enumerate(series) if idx not in excluded]
    if not filtered:
        raise RuntimeError(f"no FPS samples found in {perf_info_path}")
    return statistics.mean(filtered)


def _run_confirm_attempt(
    *,
    workspace: Path,
    artifact_dir: Path,
    launch_config: dict,
    ci_image_tag: str,
    task_id: str,
    backend_key: str,
    attempt: int,
) -> float | None:
    attempt_dir = artifact_dir / f"confirm_attempt_{attempt}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    log_file = attempt_dir / "benchmark.log"
    timeout_s = int(launch_config.get("timeout_minutes", 12)) * 60
    container_name = f"perf-confirm-{_safe_name(task_id)}-{_safe_name(backend_key)}-{int(time.time())}-{attempt}"

    hydra_args = " ".join(str(arg) for arg in launch_config.get("hydra_args") or [])
    seed = launch_config.get("seed")
    seed_arg = f"--seed {seed}" if seed is not None else ""

    subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    docker_cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "--init",
        "--stop-timeout",
        "10",
        "--entrypoint",
        "bash",
        "--gpus",
        "all",
        "--network=host",
        "--security-opt=no-new-privileges:true",
        "--ulimit",
        "nofile=65536:65536",
        "--ulimit",
        "nproc=4096:4096",
        "-e",
        "OMNI_KIT_ACCEPT_EULA=yes",
        "-e",
        "ACCEPT_EULA=Y",
        "-e",
        "OMNI_KIT_DISABLE_CUP=1",
        "-e",
        "ISAAC_SIM_HEADLESS=1",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "WARP_CACHE_PATH=/tmp/jit-cache/warp",
        "-e",
        "CUDA_CACHE_PATH=/tmp/jit-cache/nv",
        "-v",
        f"{attempt_dir}:/tmp/bench_out",
        "-v",
        f"{workspace / 'jit-cache'}:/tmp/jit-cache",
        "-v",
        f"{workspace / 'kit-cache'}:/isaac-sim/kit/cache",
        "-v",
        f"{workspace}:/workspace/isaaclab",
        ci_image_tag,
        "-c",
        "\n".join(
            [
                "set -e",
                "cd /workspace/isaaclab",
                "rm -f _isaac_sim",
                "ln -s /isaac-sim _isaac_sim",
                "./isaaclab.sh -p scripts/benchmarks/benchmark_non_rl.py "
                f"--task '{task_id}' "
                f"--num_envs {launch_config['num_envs']} "
                f"--num_frames {launch_config['num_frames']} "
                "--benchmark_backend json "
                "--output_path /tmp/bench_out "
                f"{seed_arg} "
                f"{hydra_args}",
            ]
        ),
    ]

    subprocess.run(docker_cmd, check=True)
    wait_returncode = 1
    try:
        wait = subprocess.run(
            ["timeout", str(timeout_s), "docker", "wait", container_name],
            capture_output=True,
            text=True,
            check=False,
        )
        wait_returncode = wait.returncode
        exit_code = int((wait.stdout or "1").strip() or "1") if wait_returncode == 0 else 1
    finally:
        if wait_returncode != 0:
            subprocess.run(["docker", "kill", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logs = subprocess.run(["docker", "logs", container_name], capture_output=True, check=False)
        log_file.write_bytes((logs.stdout or b"") + (logs.stderr or b""))
        subprocess.run(["docker", "kill", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["docker", "rm", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if exit_code != 0:
        print(f"[confirm] attempt {attempt} failed for {task_id}/{backend_key} (exit={exit_code})")
        return None

    outputs = sorted(attempt_dir.glob("benchmark_non_rl_*.json"))
    if not outputs:
        print(f"[confirm] attempt {attempt} produced no benchmark JSON for {task_id}/{backend_key}")
        return None
    fps = _gate_mean_fps(outputs[-1], launch_config)
    print(f"[confirm] {task_id}/{backend_key} attempt {attempt}: {fps:.1f} FPS")
    return fps


def main() -> int:
    args = _parse_args()
    with args.verdicts_file.open() as fh:
        records = json.load(fh)

    args.workspace.mkdir(parents=True, exist_ok=True)
    (args.workspace / "jit-cache" / "warp").mkdir(parents=True, exist_ok=True)
    (args.workspace / "jit-cache" / "nv").mkdir(parents=True, exist_ok=True)
    (args.workspace / "kit-cache").mkdir(parents=True, exist_ok=True)

    block_records = [record for record in records if record.get("verdict") == "BLOCK"]
    if not block_records:
        print("[confirm] no BLOCK cells to confirm")
        return 0

    for record in block_records:
        artifact_dir = Path(record["artifact_dir"])
        result_path = artifact_dir / "perf_smoke_test_result.json"
        info_path = artifact_dir / "perf_smoke_test_info.json"
        with result_path.open() as fh:
            bench_result = json.load(fh)
        launch_config = bench_result.get("launch_config") or {}
        task_id = bench_result["task_id"]
        backend_key = bench_result.get("backend_key") or bench_result.get("backend")
        attempts = [_gate_mean_fps(info_path, launch_config)]

        print(f"[confirm] confirming {task_id}/{backend_key}; initial={attempts[0]:.1f} FPS")
        for offset in range(args.reruns):
            fps = _run_confirm_attempt(
                workspace=args.workspace,
                artifact_dir=artifact_dir,
                launch_config=launch_config,
                ci_image_tag=args.ci_image_tag,
                task_id=task_id,
                backend_key=backend_key,
                attempt=offset + 2,
            )
            if fps is not None:
                attempts.append(fps)

        bench_result["confirmation_fps_attempts"] = attempts
        bench_result["confirmation_policy"] = {
            "trigger": "initial_block",
            "requested_reruns": args.reruns,
            "completed_attempts": len(attempts),
        }
        with result_path.open("w") as fh:
            json.dump(bench_result, fh, indent=2)
            fh.write("\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
