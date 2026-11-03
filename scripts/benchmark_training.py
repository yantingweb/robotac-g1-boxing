"""Short, real G1 tracking/PPO throughput experiment (not policy evaluation).

Run each configuration in a fresh process. Timings use synchronized iteration
boundaries; CUDA event spans are diagnostic, not kernel-only measurements.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import time

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-envs', type=int, default=64)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--iterations', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--task', default='Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation')
    p.add_argument('--motion', type=Path, default=Path(__file__).resolve().parents[1]/'data/Quick_Jab_train.npz')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if min(a.num_envs, a.iterations) < 1 or a.warmup < 1:
        p.error('positive environment/iteration counts and at least one warmup required')
    if a.output.exists():
        p.error('output already exists; use a fresh filename')
    import mujoco
    # Existing project compatibility shim, applied only in this process.
    if not hasattr(mujoco.mjtEnableBit, 'mjENBL_MULTICCD'):
        mujoco.mjtEnableBit.mjENBL_MULTICCD = 1 << 6
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    torch.manual_seed(a.seed)
    ec = load_env_cfg(a.task)
    ec.scene.num_envs = a.num_envs
    ec.seed = a.seed
    ec.commands['motion'].motion_file = str(a.motion.resolve())
    rc = load_rl_cfg(a.task)
    rc.seed = a.seed
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(ec, device='cuda:0'), clip_actions=rc.clip_actions)
    try:
        cfg = asdict(rc)
        runner = MjlabOnPolicyRunner(env, cfg, log_dir=None, device='cuda:0')
        alg = runner.alg
        alg.train_mode()
        obs = env.get_observations().to('cuda:0')
        rows = []
        for iteration in range(a.warmup+a.iterations):
            if iteration == a.warmup:
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            events = {k: [] for k in ('inference', 'environment', 'storage', 'returns', 'update')}
            def timed(name, fn):
                x, y = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                x.record()
                with torch.profiler.record_function(name):
                    result = fn()
                y.record()
                events[name].append((x, y))
                return result
            rewards_sum = 0.0
            done_sum = 0
            with torch.inference_mode():
                for _ in range(cfg['num_steps_per_env']):
                    actions = timed('inference', lambda: alg.act(obs))
                    obs, rewards, dones, extras = timed('environment', lambda: env.step(actions))
                    timed('storage', lambda: alg.process_env_step(obs, rewards, dones, extras))
                    # Keep summaries on-device until end of collection.
                    rewards_sum = rewards_sum + rewards.sum()
                    done_sum = done_sum + dones.sum()
                timed('returns', lambda: alg.compute_returns(obs))
            losses = timed('update', alg.update)
            torch.cuda.synchronize()
            wall = time.perf_counter()-start
            row = dict(iteration=iteration, measured=iteration >= a.warmup,
                       wall_s=wall, transitions=a.num_envs*cfg['num_steps_per_env'],
                       cuda_span_ms={k: sum(x.elapsed_time(y) for x,y in v) for k,v in events.items()},
                       mean_step_reward=float(rewards_sum)/(a.num_envs*cfg['num_steps_per_env']),
                       terminations=int(done_sum), losses={k: float(v) for k,v in losses.items()})
            rows.append(row)
            print(json.dumps(row), flush=True)
        measured = [r for r in rows if r['measured']]
        versions = {}
        for package in ('torch','mjlab','mujoco','mujoco-warp','warp-lang','rsl-rl-lib'):
            try: versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError: versions[package] = 'not registered'
        result = dict(schema_version=1, task=a.task, num_envs=a.num_envs, seed=a.seed,
            warmup=a.warmup, iterations=a.iterations, gpu=torch.cuda.get_device_name(),
            platform=platform.platform(), versions=versions, torch_threads=torch.get_num_threads(),
            motion_sha256=hashlib.sha256(a.motion.read_bytes()).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            agent_config=cfg, rows=rows,
            transitions_per_s=sum(r['transitions'] for r in measured)/sum(r['wall_s'] for r in measured),
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
            limitations=['Random initialization; short training throughput only, not convergence or policy quality.',
             'CUDA spans include stream idle/host submission gaps; not pure GPU kernel time.',
             'PyTorch memory excludes Warp and other non-PyTorch allocations.',
             'Different environment counts change PPO batch size; configurations are not learning-equivalent.',
             'Current upstream tracking task; historical boxing reward patches/checkpoints are not loaded.'])
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    finally:
        env.close()


if __name__ == '__main__':
    main()
