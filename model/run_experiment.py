"""Run a model on explicitly selected local datasets: train -> best -> evaluation.

Only artifact organization is local. No custom training loop, loss or optimizer.
Use --prepare-only to create readable folders without consuming the GPU.
"""
import argparse
import csv
from copy import deepcopy
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from artifacts import collect_checkpoints, write_batch_report, write_json, write_loss_curves, attach_loss_section, loss_section

ROOT = Path(__file__).resolve().parents[1]
UNSTARTED = ('planned', 'waiting_for_data', 'data_failed')


def validated_dataset(dataset):
    """Verify a completed local simulation collection, regardless of its size."""
    manifest = json.loads((dataset / 'collection_manifest.json').read_text())
    report = json.loads((dataset / 'validation_report.json').read_text())
    total = len(manifest['episodes'])
    if total < 2 or report['episodes'] != total or not report['full_video_decode']:
        raise ValueError(f'{dataset.name}: need>=2 fully decoded, validated episodes')
    if manifest.get('task_variant') and report.get('physics_replay', {}).get('episodes') != total:
        raise ValueError(f'{dataset.name}: task-variant episodes need complete physics replay')
    return manifest


def dataset_path(value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if len(path.parts) == 1:
        return (ROOT / 'data' / path).resolve()
    return (ROOT / path).resolve()


def configuration(model, dataset, run, args):
    config_path = getattr(args, 'config', None) or ROOT / f'model/configs/{model}.json'
    settings = json.loads(Path(config_path).read_text())
    if settings['policy']['type'] != model:
        raise ValueError('Configuration policy type differs from --model')
    settings['dataset'].update(root=str(dataset), repo_id=f'local/{dataset.name}', eval_split=.2)
    settings['dataset'].pop('episodes', None)
    settings['output_dir'] = str(run / 'metadata/trainer')
    settings['job_name'] = f'{model}_{dataset.name}'
    if args.steps is not None:
        settings['steps'] = args.steps
    if args.batch_size is not None:
        settings['batch_size'] = args.batch_size
    freq = min(args.checkpoint_every or settings['save_freq'], settings['steps'])
    settings.update(save_freq=freq, eval_steps=math.gcd(freq, settings['steps']),
                    env_eval_freq=0, save_checkpoint=True)
    settings.setdefault('wandb', {})['enable'] = False
    if model == 'smolvla':
        settings['policy']['path'] = str(ROOT / 'checkpoints/smolvla_base')
        settings['policy']['vlm_model_name'] = str(ROOT / 'checkpoints/smolvla_vlm')
    return settings


def launch(command, log_path, run, training=False):
    """Run one child group; terminate workers on interrupt, never leave orphan GPU jobs."""
    with log_path.open('a') as log:
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='2')
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=env)
        last_sample = 0.
        last_plot = 0.
        try:
            while child.poll() is None:
                if training:
                    collect_checkpoints(run)
                    if time.monotonic() - last_plot >= 60:
                        summary = write_loss_curves(run)
                        attach_loss_section(run/'README.md', loss_section(summary))
                        last_plot = time.monotonic()
                if time.monotonic() - last_sample >= 5:
                    try:
                        sample = subprocess.run(['nvidia-smi', '--id=0',
                            '--query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
                        values = [value.strip() for value in sample.stdout.strip().split(',')]
                        if len(values) == 4:
                            path = run/'metadata/gpu_usage.csv'
                            new = not path.exists()
                            with path.open('a', newline='') as gpu_log:
                                writer = csv.writer(gpu_log)
                                if new:
                                    writer.writerow(['time', 'stage', 'utilization_percent', 'used_mib', 'free_mib', 'temperature_c'])
                                writer.writerow([datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                                                 'training' if training else 'evaluation', *values])
                    except (OSError, subprocess.SubprocessError):
                        pass  # Telemetry failure must not abort an otherwise healthy model job.
                    last_sample = time.monotonic()
                time.sleep(1)
            if training:
                collect_checkpoints(run)
            if child.returncode:
                raise RuntimeError(f'Child exited {child.returncode}; see {log_path}')
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


def run_dataset(run, args):
    state_file = run / 'metadata/run.json'
    state = json.loads(state_file.read_text())
    dataset = Path(state['dataset'])
    settings = json.loads((run / 'metadata/train_config.json').read_text())

    def save(status, error=None):
        state.update(status=status, updated=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        if error is not None:
            state['error'] = str(error)
        else:
            state.pop('error', None)
        write_json(state_file, state)
        write_batch_report(run.parent)
        print(f'{run.name}: {status}', flush=True)

    try:
        validated_dataset(dataset)
        if args.eval_only:
            if not (run / 'checkpoints/best/pretrained_model/model.safetensors').exists():
                raise ValueError('No saved best model to evaluate')
        else:
            if state['status'] not in UNSTARTED and not args.resume:
                raise ValueError('Run exists: use --resume or --eval-only; nothing overwritten')
            if args.resume and state['status'] not in UNSTARTED:
                original = run / 'metadata/trainer/checkpoints/last/pretrained_model/train_config.json'
                if not original.exists():
                    original = run / 'checkpoints/latest/pretrained_model/train_config.json'
                if not original.exists():
                    raise ValueError('No complete official checkpoint to resume')
                progress = original.parent.parent / 'training_state/training_step.json'
                completed = json.loads(progress.read_text())['step']
                command = [sys.executable, '-m', 'lerobot.scripts.lerobot_train',
                           f'--config_path={original}', '--resume=true',
                           f'--output_dir={run / "metadata/trainer"}']
            else:
                completed = 0
                command = [sys.executable, '-m', 'lerobot.scripts.lerobot_train',
                           f'--config_path={run / "metadata/train_config.json"}']
            if completed < settings['steps']:
                save('training')
                launch(command, run / 'metadata/train.log', run, training=True)
            latest = json.loads((run / 'metadata/latest.json').read_text())
            if latest['step'] != settings['steps']:
                raise ValueError('Latest checkpoint is not the planned final step')
            # Only remove staging after official training has exited and both
            # promoted copies exist. Keep all metadata and the raw training log.
            for name in ('latest', 'best'):
                if not (run / f'checkpoints/{name}/pretrained_model/model.safetensors').is_file():
                    raise ValueError(f'Missing promoted {name} checkpoint')
            import shutil
            staging = run / 'metadata/trainer/checkpoints'
            if staging.exists():
                shutil.rmtree(staging)
        evaluation = run / 'metadata/evaluation'
        if evaluation.exists():
            raise ValueError('Evaluation already exists; preserve it and choose a new run for a new test')
        # Fixed test seeds independent of every collection seed and held-out
        # demonstration. No best-checkpoint selection on these test episodes.
        manifest = json.loads((dataset / 'collection_manifest.json').read_text())
        collected = {e['seed'] for e in manifest['episodes']}
        tested = set(range(state['eval_seed'], state['eval_seed'] + state['eval_episodes']))
        if collected & tested:
            raise ValueError('Test seeds overlap collection seeds')
        save('evaluating')
        command = [sys.executable, str(ROOT / 'model/play/policy.py'),
                   '--checkpoint', str(run / 'checkpoints/best/pretrained_model'),
                   '--dataset', str(dataset), '--episodes', str(state['eval_episodes']),
                   '--seed', str(state['eval_seed']), '--max-steps', str(state['max_steps']),
                   '--video-count', '3', '--video-dir', str(run / 'videos'), '--stable-finish',
                   '--output', str(evaluation)]
        write_json(run / 'metadata/eval_command.json', command)
        launch(command, run / 'metadata/eval.log', run)
        results = json.loads((evaluation / 'results.json').read_text())
        if len(results['episodes']) != state['eval_episodes']:
            raise ValueError('Incomplete evaluation')
        if len(list((run / 'videos').glob('*.mp4'))) != 3:
            raise ValueError('Expected exactly three evaluation videos')
        save('completed')
    except BaseException as exc:
        save('interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed', exc)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['act', 'diffusion', 'smolvla'], required=True)
    p.add_argument('--datasets', '--dataset', dest='datasets', nargs='+', required=True,
                   help='Dataset names under data/, project-relative paths, or absolute paths')
    p.add_argument('--config', type=Path, help='Optional official training config template')
    p.add_argument('--date', default=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d'))
    p.add_argument('--output', type=Path, help='Override model/date folder; never silently create train_02')
    p.add_argument('--steps', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--checkpoint-every', type=int)
    p.add_argument('--eval-episodes', type=int, default=10)
    p.add_argument('--eval-seed', type=int, default=10000)
    p.add_argument('--max-steps', type=int, default=3600)
    p.add_argument('--prepare-only', action='store_true', help='Create folders/configs only; no GPU work')
    p.add_argument('--dry-run', action='store_true', help='Print layout/configs; write nothing')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--eval-only', action='store_true')
    args = p.parse_args()
    for name in ('steps', 'batch_size', 'checkpoint_every', 'max_steps'):
        value = getattr(args, name)
        if value is not None and value < 1:
            p.error(f'{name} must be positive')
    if args.eval_episodes < 3:
        p.error('eval-episodes must be>=3 to save three real evaluation videos')
    if not (len(args.date) == 8 and args.date.isdigit()):
        p.error('date must be YYYYMMDD')
    batch = (args.output or ROOT / 'outputs' / f'{args.model}{args.date}').resolve()
    datasets = [dataset_path(value) for value in args.datasets]
    if len({path.name for path in datasets}) != len(datasets):
        p.error('Dataset directory names must be unique within an experiment batch')
    planned = []
    for dataset in datasets:
        run = batch / dataset.name
        settings = configuration(args.model, dataset, run, args)
        state = dict(status='planned', model=args.model, dataset=str(dataset),
                     steps=settings['steps'], eval_episodes=args.eval_episodes, eval_seed=args.eval_seed,
                     max_steps=args.max_steps, created=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        planned.append((run, settings, state))
    if args.dry_run:
        print(json.dumps([dict(output=str(run), config=s, evaluation=state) for run, s, state in planned], ensure_ascii=False, indent=2))
        return
    batch.mkdir(parents=True, exist_ok=True)
    lock_folder = ROOT / '.cache/experiment_locks'
    lock_folder.mkdir(parents=True, exist_ok=True)
    import hashlib
    lock_path = lock_folder / (hashlib.sha256(str(batch).encode()).hexdigest() + '.lock')
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error('This experiment batch is already active')
        # Preflight every selected run before starting any GPU work.
        for run, settings, state in planned:
            state_file = run / 'metadata/run.json'
            if state_file.exists():
                old = json.loads(state_file.read_text())
                old_config = json.loads((run / 'metadata/train_config.json').read_text())
                expected = deepcopy(state); expected.pop('created'); expected.pop('status')
                actual = {k: old[k] for k in expected}
                can_replan = old['status'] in UNSTARTED and not (run/'metadata/train.log').exists() and not any((run/'checkpoints').iterdir())
                if (expected != actual or old_config != settings) and not can_replan:
                    p.error(f'Existing settings differ: {run}; choose a new --output')
                if args.eval_only and old['status'] == 'completed':
                    p.error(f'Evaluation already completed: {run}; nothing overwritten')
                if old['status'] not in UNSTARTED and not (args.resume or args.eval_only or args.prepare_only):
                    p.error(f'Run exists: {run}; use --resume/--eval-only')
            elif run.exists() and any(run.iterdir()):
                p.error(f'Unmanaged nonempty folder: {run}')
            elif args.resume or args.eval_only:
                p.error(f'No existing run: {run}')
            if not args.prepare_only:
                try:
                    validated_dataset(Path(state['dataset']))
                except (OSError, ValueError, KeyError) as exc:
                    p.error(f'Dataset not ready: {run.name}: {exc}')
        for run, settings, state in planned:
            for folder in ('checkpoints', 'videos', 'metadata'):
                (run / folder).mkdir(parents=True, exist_ok=True)
            state_file = run / 'metadata/run.json'
            if not state_file.exists() or json.loads(state_file.read_text())['status'] in UNSTARTED:
                write_json(run / 'metadata/train_config.json', settings)
                write_json(state_file, state)
        write_batch_report(batch)
        if args.prepare_only:
            print(f'Prepared only; no training/evaluation started: {batch}')
            return
        for run, _, _ in planned:
            old = json.loads((run / 'metadata/run.json').read_text())
            if args.resume and old['status'] == 'completed':
                continue
            run_dataset(run, args)


if __name__ == '__main__':
    # SIGTERM follows the same child cleanup path as Ctrl+C.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    main()
