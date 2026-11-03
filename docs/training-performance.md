# G1 tracking training performance

This experiment asks how environment parallelism affects short-run training
throughput on one 8 GB laptop GPU. It runs real MuJoCo-Warp environment steps,
policy inference, return computation and PPO updates. It does not replace the
historical boxing experiment or measure learned policy quality.

## Reproduce

Use Linux/WSL2 with an NVIDIA GPU and a working mjlab installation. The measured
package versions are saved in each raw JSON file. This experiment uses the
current upstream `Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation` task and
the repository's `data/Quick_Jab_train.npz`. Do not apply the historical boxing
patch files to this environment: they target an earlier framework API.

```bash
MUJOCO_GL=egl python scripts/benchmark_training.py \
  --num-envs 64 --seed 42 --warmup 3 --iterations 8 \
  --output outputs/performance/n64_s42.json
```

Repeat with 256 and 1024 environments, and seeds 42 and 43, sequentially in
fresh processes. Keep other GPU workloads idle. Output filenames cannot be
overwritten. Then summarize:

```bash
python scripts/summarize_performance.py outputs/performance \
  --csv outputs/performance/summary.csv
```

The sequential matrix can also be launched with
`python scripts/run_performance_sweep.py --output outputs/performance`.
For the thread comparison use `--envs 1024 --threads 1` with a different output
directory. The launcher fails on an existing result instead of mixing runs.

For a controlled CPU thread-pool comparison, set `OMP_NUM_THREADS=1` before
starting the same command, and use a separate output directory. Compare only
matching environment counts and seeds. This setting is an experimental variable,
not a guaranteed optimization.

## Measurement contract

- Each iteration collects 24 steps per environment and performs the configured
  PPO update. Three complete warmup iterations are excluded. Eight measured
  iterations per seed are a short performance probe, not a convergence study.
- Throughput is total measured environment transitions divided by total measured
  wall time. Independent seeds are summarized with mean and sample standard
  deviation. Iterations within a run are not treated as independent replicates.
- CUDA is synchronized at iteration boundaries. CUDA events span inference,
  environment stepping, storage, return calculation and policy updates. These
  spans include stream idle time and CPU submission gaps; they are not pure
  kernel execution times and do not isolate physics from all environment logic.
- Collection includes reward/done reductions and timing instrumentation. Video,
  external logging, checkpoint writing and runner logging are excluded. This is
  an instrumented training-loop benchmark, not a measurement of the unmodified
  `runner.learn()` command. No historical throughput baseline is inferred.
- PyTorch peak allocated/reserved memory excludes allocations made by Warp,
  MuJoCo and other GPU users. Do not label it total GPU memory usage.
- Increasing environment count also increases PPO batch size. Higher transition
  throughput does not establish faster convergence, equal reward, or better
  time-to-solution. Reward/loss/termination summaries are saved for inspection,
  not used as a final task-quality evaluation.
- This laptop run does not lock GPU clocks or control all desktop activity.
  The fixed run order and only two seeds limit conclusions about small changes;
  do not extrapolate to server GPUs or claim statistically proven speedups.
- The existing MuJoCo 3.10 MULTICCD compatibility shim is applied in the process
  without modifying installed packages. Its presence is part of this setup;
  these measurements do not certify simulator physical accuracy.

Raw JSON records contain package versions, GPU model, task, seed, motion/script
SHA256, algorithm configuration and every iteration. Failed attempts stay in
local logs and are not converted into successful results.

## Results

See [measured results and raw records](performance/2026-11-03/RESULTS.md).
