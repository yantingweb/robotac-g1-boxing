"""Summarize independent short training runs; excludes smoke outputs."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def summarize(paths):
    runs = [json.loads(p.read_text()) for p in paths]
    if not runs:
        raise ValueError('No runs found')
    # Thread count and batch size are intentional independent variables.
    common = ('task', 'gpu', 'versions', 'motion_sha256', 'warmup', 'iterations', 'agent_config')
    ref = runs[0]
    for r in runs:
        for k in common:
            left, right = r[k], ref[k]
            if k == 'agent_config':
                left = {k:v for k,v in left.items() if k != 'seed'}
                right = {k:v for k,v in right.items() if k != 'seed'}
            if left != right:
                raise ValueError('Incompatible runs: '+k)
        if not math.isfinite(r['transitions_per_s']) or r['transitions_per_s'] <= 0:
            raise ValueError('Invalid throughput')
    groups = {}
    seen = set()
    for r in runs:
        key = (r['num_envs'], r['torch_threads'])
        identity = (*key, r['seed'])
        if identity in seen:
            raise ValueError('Duplicate configuration and seed')
        seen.add(identity)
        groups.setdefault(key, []).append(r)
    rows = []
    for (n,t), group in sorted(groups.items()):
        rates = [r['transitions_per_s'] for r in group]
        measured = [x for r in group for x in r['rows'] if x['measured']]
        rows.append(dict(num_envs=n, torch_threads=t, repeats=len(group),
            seeds=','.join(str(r['seed']) for r in group),
            transitions_per_s_mean=statistics.mean(rates),
            transitions_per_s_stdev=statistics.stdev(rates) if len(rates)>1 else 0,
            iteration_s_mean=statistics.mean(x['wall_s'] for x in measured),
            peak_torch_allocated_mib=max(r['peak_torch_allocated_bytes'] for r in group)/2**20,
            environment_span_ms_mean=statistics.mean(x['cuda_span_ms']['environment'] for x in measured),
            inference_span_ms_mean=statistics.mean(x['cuda_span_ms']['inference'] for x in measured),
            update_span_ms_mean=statistics.mean(x['cuda_span_ms']['update'] for x in measured)))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--csv', type=Path, required=True)
    a = p.parse_args()
    rows = summarize(sorted(a.directory.glob('n*_s*.json')))
    a.csv.parent.mkdir(parents=True, exist_ok=True)
    with a.csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
