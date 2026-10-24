"""Prepare 200 demonstrations, check official SmolVLA interfaces, fine-tune, then evaluate."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'pipeline.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('This pipeline is already running')
    if (output/'run.json').exists():
        parser.error('Training already started in this run; inspect run.json instead of starting a duplicate')
    config = ROOT/'model/configs/smolvla.json'
    state = {'pid': os.getpid(), 'config': str(config), 'started': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}

    def save(status):
        state.update(status=status, updated=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        temp = output/'pipeline_status.json.tmp'
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')
        temp.replace(output/'pipeline_status.json')
        print(status, flush=True)

    def run(command, log_name, status):
        save(status)
        with (output/log_name).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state.update(child_pid=child.pid, current_command=command)
            save(status)
            code = child.wait()
        if code:
            raise RuntimeError(f'{status} failed with code {code}; see {log_name}')
        state.pop('child_pid', None)

    try:
        dataset = ROOT/'data/official_tabletop_200'
        if not (output/'data_status.json').exists() and not (dataset/'validation_report.json').exists():
            with (output/'prepare.log').open('w') as log:
                child = subprocess.Popen([sys.executable, str(ROOT/'myscript/expand_dataset.py'), '--run', str(output)],
                                         cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                state['preparation_pid'] = child.pid
        assets_file = output/'assets_status.json'
        assets = json.loads(assets_file.read_text()) if assets_file.exists() else {}
        if assets.get('status') != 'completed':
            run([sys.executable, str(ROOT/'model/download_smolvla.py'), '--run', str(output)], 'download.log', 'downloading')
        smoke = output/'interface_check'
        ready = smoke/'READY.json'
        if not ready.exists():
            checkpoint = smoke/'train/checkpoints/last/pretrained_model'
            if not (checkpoint/'model.safetensors').exists():
                run([sys.executable, str(ROOT/'model/train/smolvla.py'), '--steps=8', '--save_freq=8',
                     '--eval_steps=0', '--log_freq=1', '--dataset.root=data/official_tabletop_50',
                     '--dataset.repo_id=local/so101_official_tabletop', f'--output_dir={smoke/"train"}'],
                    'interface_train.log', 'checking_train_interface')
            if not (smoke/'eval/results.json').exists():
                run([sys.executable, str(ROOT/'model/play/policy.py'), '--checkpoint', str(checkpoint),
                     '--dataset', str(ROOT/'data/official_tabletop_50'), '--episodes', '1', '--seed', '10000',
                     '--max-steps', '16', '--output', str(smoke/'eval')], 'interface_eval.log', 'checking_sim_interface')
            check = json.loads((smoke/'eval/results.json').read_text())
            if len(check['episodes']) != 1 or check['episodes'][0]['steps'] != 16:
                raise ValueError('Incomplete simulator interface check')
            ready.write_text(json.dumps({'train_save_load_sim_passed': True,
                                         'note': 'Interface check only; formal run starts from official base weights.'}, indent=2)+'\n')
        while not (dataset/'validation_report.json').exists():
            progress_file = output/'data_status.json'
            if progress_file.exists():
                progress = json.loads(progress_file.read_text())
                if progress['status'] == 'failed':
                    raise RuntimeError(f'Data preparation failed: {progress.get("error")}')
                if progress['status'] == 'completed':
                    raise RuntimeError('Preparation reported completed but validated data is missing')
                proc = Path(f"/proc/{progress['pid']}/cmdline")
                try: live = str(ROOT/'myscript/expand_dataset.py').encode() in proc.read_bytes().split(b'\0')
                except FileNotFoundError: live = False
                if not live:
                    raise RuntimeError('Data preparation was interrupted; inspect data_status.json and preserve partial data')
                state['data_stage'] = progress['status']
            manifest = ROOT/'data/official_tabletop_extra150/collection_manifest.json'
            try: state['new_episodes_collected'] = len(json.loads(manifest.read_text())['episodes'])
            except (FileNotFoundError, json.JSONDecodeError): pass
            save('waiting_for_validated_data')
            time.sleep(15)
        report = json.loads((dataset/'validation_report.json').read_text())
        manifest = json.loads((dataset/'collection_manifest.json').read_text())
        if report['episodes'] != 200 or len(manifest['episodes']) != 200:
            raise ValueError('Expected exactly 200 validated demonstrations')
        if len({ep['seed'] for ep in manifest['episodes']}) != 200:
            raise ValueError('Repeated collection seeds')
        state['validated_episodes'] = 200
        state['validated_frames'] = report['frames']
        state['data_stage'] = 'completed'
        state['new_episodes_collected'] = sum(ep.get('source_root') == str(ROOT/'data/official_tabletop_extra150')
                                              for ep in manifest['episodes'])
        # Always load the original pretrained base; the tiny interface run is never a warm-start source.
        run([sys.executable, str(ROOT/'model/train_and_eval.py'), '--config', str(config),
             '--output', str(output), '--episodes', '10', '--seed', '10000'], 'train_eval_launcher.log', 'training_and_evaluating')
        save('completed')
    except Exception as exc:
        state['error'] = str(exc)
        save('failed')
        raise


if __name__ == '__main__':
    main()
