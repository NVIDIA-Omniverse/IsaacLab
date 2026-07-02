# 3.0 vs 2.3.2 Regression Evidence

This note is the shareable checklist/report shell for the follow-up requested by
the team after the PhysX FPS graph. The actual numeric table is generated from
CI artifacts by:

```bash
python3 tools/perf_smoke_test/regression_evidence_pack.py \
  --input isaaclab_2_3_2=regression-evidence/isaaclab_2_3_2 \
  --input current_fork=regression-evidence/current_fork
```

The manual workflow that produces those artifacts is:

```text
Performance Smoke - Regression Evidence Pack
```

## What Gets Collected

- Repeated unprofiled `benchmark_non_rl.py` runs for IsaacLab 2.3.2 and the
  current 3.0 fork using matching task, env-count, seed, and warm-up exclusion.
- One shorter Nsight Systems profile per release/task. The workflow uploads the
  `.nsys-rep` trace and `nsys stats` CSVs (`cuda_gpu_kern_sum`, `osrt_sum`,
  `cuda_api_sum`) as CI artifacts for manual Google Drive upload.
- VRAM from benchmark runtime measurements when present, plus `nvidia-smi`
  samples.
- System RAM usage from `docker stats` samples, summarized as mean and peak MB.
- CPU evidence for Antoine's question: `lscpu`, `/proc/meminfo`, `pidstat`
  samples when available, Docker CPU percentage, and host PID metadata.
- One perception workload:
  `Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0` with cameras enabled.

## Why Cartpole Needs A CPU Note

Cartpole is the most overhead-bound task in this comparison. If the RTX PRO 6000
runner is behind Antoine's laptop on Cartpole, the first thing to verify is the
runner CPU model, physical core count, CPU governor/frequency, Docker CPU
limits, and whether one host thread is saturated during the step loop. The new
workflow collects those signals, so the next report can separate:

- a real 3.0-vs-2.3.2 fork regression, visible within the same runner;
- a shared runner CPU ceiling that depresses both releases' absolute FPS;
- profiling overhead from the nsys pass, which is kept separate from the
  unprofiled statistical samples.

## Artifact Layout

After the workflow completes, download the `regression-evidence-<run_id>`
artifact. Important paths:

- `host/lscpu.txt`: CPU model and physical topology.
- `host/nvidia-smi-summary.csv`: GPU, driver, CUDA, and total VRAM.
- `<label>/<task>/sample_*/benchmark_output.json`: raw benchmark JSON.
- `<label>/<task>/sample_*/docker_stats.jsonl`: system RAM and Docker CPU
  samples.
- `<label>/<task>/sample_*/nvidia_smi_samples.csv`: GPU utilization and VRAM
  samples.
- `<label>/<task>/sample_*/pidstat.log`: process-level CPU/memory samples when
  `pidstat` is available on the runner.
- `<label>/<task>/nsys/nsys_trace.nsys-rep`: trace for Nsight Systems.
- `<label>/<task>/nsys/nsys_*.csv`: trace summary reports.
- `regression_evidence_summary.json`: machine-readable summary used by the
  chart script.
- `3_0_vs_2_3_2_regression_evidence.md`: generated markdown summary.

## Slack Reply Skeleton

Once the workflow artifact is downloaded and the Drive links are available:

```text
I collected a matched 2.3.2 vs current-fork evidence pack on the RTX PRO 6000
runner. The artifact includes repeated unprofiled samples for FPS/std-dev,
VRAM and system-RAM usage, host CPU details, Docker CPU/RAM samples, and one
nsys trace per release/task. The traces have been uploaded here: <Drive link>.

Cartpole remains the key CPU/host-overhead signal. The runner CPU is <CPU from
lscpu>, with <physical cores> physical cores. Compared with Antoine's laptop
number, the absolute Cartpole FPS looks <CPU-bound / not CPU-bound> because
<pidstat/docker-stats/nsys osrt evidence>. The within-run fork-vs-2.3.2 delta
is still <delta>, so <shared CPU ceiling does/does not> explain the full gap.

We also added Shadow Vision Benchmark as a representative perception workload.
Its fork-vs-2.3.2 result is <delta>, with VRAM <values> and system RAM <values>.
```
