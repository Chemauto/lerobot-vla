"""Measure ACT batch capacity with synthetic tensors; never save learned weights.

Uses the official policy, shapes from the actual dataset, BF16 forward/backward,
AdamW and clipping. This excludes video decoding; the real trainer still needs
enough dataloader throughput. Each candidate runs in its own CUDA process.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def candidate(args):
    import torch
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.factory import make_policy
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    meta = LeRobotDatasetMetadata(f'local/{args.dataset.name}', root=args.dataset)
    cfg = ACTConfig(device='cuda', chunk_size=50, n_action_steps=10, push_to_hub=False)
    policy = make_policy(cfg, ds_meta=meta).train()
    optimizer = cfg.get_optimizer_preset().build(policy.get_optim_params())
    batch = {key: torch.zeros((args.batch, *feature.shape), device='cuda')
             for key, feature in cfg.input_features.items()}
    batch['action'] = torch.zeros((args.batch, cfg.chunk_size, 6), device='cuda')
    batch['action_is_pad'] = torch.zeros((args.batch, cfg.chunk_size), dtype=torch.bool, device='cuda')
    torch.cuda.reset_peak_memory_stats()
    times = []
    for step in range(10):
        torch.cuda.synchronize()
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = policy(batch)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite synthetic loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.)
        optimizer.step()
        torch.cuda.synchronize()
        if step >= 4:
            times.append(time.monotonic() - started)
    free, total = torch.cuda.mem_get_info()
    result = dict(status='ok', batch=args.batch, seconds_per_step=sum(times)/len(times),
                  samples_per_second=args.batch*len(times)/sum(times),
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                  free_mib=free/2**20, total_mib=total/2**20)
    print('BENCHMARK_RESULT ' + json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=ROOT/'data/one_fixed_200')
    p.add_argument('--batches', type=int, nargs='+', default=[8, 12, 16])
    p.add_argument('--batch', type=int, help=argparse.SUPPRESS)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.dataset = args.dataset.resolve()
    if args.batch is not None:
        try:
            candidate(args)
        except Exception as exc:
            # A separate process releases all CUDA allocations even after OOM.
            print('BENCHMARK_RESULT ' + json.dumps(dict(status='failed', batch=args.batch, error=str(exc))), flush=True)
            raise
        return
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for batch in args.batches:
        command = [sys.executable, str(Path(__file__).resolve()), '--dataset', str(args.dataset),
                   '--batch', str(batch), '--output', str(args.output)]
        with (args.output/f'batch{batch}.log').open('w') as log:
            code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        text = (args.output/f'batch{batch}.log').read_text(errors='replace')
        values = [json.loads(line.removeprefix('BENCHMARK_RESULT ')) for line in text.splitlines()
                  if line.startswith('BENCHMARK_RESULT ')]
        result = values[-1] if values else dict(status='failed', batch=batch, exit_code=code)
        results.append(result)
        (args.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
