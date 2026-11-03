"""Run a sequential environment-count sweep with fresh processes."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--envs', type=int, nargs='+', default=[64, 256, 1024])
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    p.add_argument('--threads', type=int)
    p.add_argument('--output', type=Path, default=Path('outputs/performance'))
    a = p.parse_args()
    if a.threads is not None and a.threads < 1:
        p.error('threads must be positive')
    root = Path(__file__).resolve().parents[1]
    a.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MUJOCO_GL='egl')
    if a.threads is not None:
        env['OMP_NUM_THREADS'] = str(a.threads)
    for seed in a.seeds:
        for n in a.envs:
            result = a.output/f'n{n}_s{seed}.json'
            if result.exists():
                raise FileExistsError(result)
            command = [sys.executable, '-u', str(root/'scripts/benchmark_training.py'),
                       '--num-envs', str(n), '--seed', str(seed), '--warmup', '3',
                       '--iterations', '8', '--output', str(result.resolve())]
            print('Running', n, seed, flush=True)
            with result.with_suffix('.log').open('x') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
