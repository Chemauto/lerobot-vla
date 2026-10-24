"""Keep the original demonstrations, collect the remainder, and merge using LeRobot APIs."""

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT/'data/official_tabletop_50')
    parser.add_argument('--additional', type=Path, default=ROOT/'data/official_tabletop_extra150')
    parser.add_argument('--output', type=Path, default=ROOT/'data/official_tabletop_200')
    parser.add_argument('--total', type=int, default=200)
    parser.add_argument('--run', type=Path, required=True, help='Logs and preparation status directory')
    args = parser.parse_args()
    base, extra, output, run = [p.resolve() for p in (args.base, args.additional, args.output, args.run)]
    if output.exists():
        parser.error(f'Output already exists: {output}; no files replaced')
    if output.with_name(output.name+'.building').exists():
        parser.error('An incomplete merge exists; inspect it before retrying')
    original = json.loads((base/'collection_manifest.json').read_text())
    needed = args.total - len(original['episodes'])
    if needed < 1:
        parser.error('total must exceed the original episode count')
    seed = max(e['seed'] for e in original['attempts']) + 1
    run.mkdir(parents=True, exist_ok=True)
    state = {'pid': os.getpid(), 'base': str(base), 'additional': str(extra), 'output': str(output),
             'target_episodes': args.total, 'additional_episodes': needed, 'first_new_seed': seed}

    def save(status):
        state.update(status=status, updated=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        temp = run/'data_status.json.tmp'
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')
        temp.replace(run/'data_status.json')
        print(status, flush=True)

    try:
        save('collecting')
        if not extra.exists():
            command = [sys.executable, str(ROOT/'myscript/collect_official.py'),
                       '--episodes', str(needed), '--max-attempts', str(needed*3),
                       '--seed', str(seed), '--root', str(extra), '--preview']
            state['collection_command'] = command
            with (run/'collect150.log').open('w') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                state['child_pid'] = child.pid
                save('collecting')
                code = child.wait()
            if code:
                raise RuntimeError(f'Collector exited {code}; retained episodes preserved in {extra}')
        newer = json.loads((extra/'collection_manifest.json').read_text())
        info = json.loads((extra/'meta/info.json').read_text())
        if len(newer['episodes']) != needed or info['total_episodes'] != needed:
            raise ValueError('Additional dataset is incomplete; refusing to merge or train')
        for key in ('env_repo', 'env_revision', 'fps', 'initial_joints', 'state_units', 'alignment', 'physics_overrides'):
            if original[key] != newer[key]:
                raise ValueError(f'Source environments differ: {key}')
        for name in ('official_teacher.py', 'official_env.py', 'official_ik.py'):
            if original['sources_sha256'][name] != newer['sources_sha256'][name]:
                raise ValueError(f'Expert source changed: {name}')
        config = 'collector_sources/configs/official_tabletop.json'
        if (base/config).read_bytes() != (extra/config).read_bytes():
            raise ValueError('Camera/environment configurations differ')
        seeds = [e['seed'] for m in (original, newer) for e in m['episodes']]
        if len(set(seeds)) != args.total or any(10000 <= s <= 10009 for s in seeds):
            raise ValueError('Repeated seeds or evaluation-seed leakage')

        save('merging')
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.dataset_tools import merge_datasets
        import numpy as np
        import pandas as pd

        building = output.with_name(output.name+'.building')
        datasets = [LeRobotDataset('local/so101_official_tabletop', root=p, video_backend='pyav') for p in (base, extra)]
        merged = merge_datasets(datasets, 'local/so101_official_tabletop_200', output_dir=building,
                                concatenate_videos=False, concatenate_data=False)
        if merged.meta.total_episodes != args.total:
            raise ValueError('Merged episode count differs from target')
        manifest = deepcopy(original)
        manifest.update(episodes=[], attempts=deepcopy(original['attempts'])+deepcopy(newer['attempts']),
                        preparation='Original episodes first, then newly collected episodes; official merge_datasets',
                        source_datasets=[])
        for source, data in ((base, original), (extra, newer)):
            manifest['source_datasets'].append({'root': str(source), 'episodes': len(data['episodes']),
                                                'sources_sha256': data['sources_sha256'], 'versions': data['versions']})
            for episode in data['episodes']:
                record = deepcopy(episode)
                record.update(source_root=str(source), source_episode_index=episode['episode_index'],
                              episode_index=len(manifest['episodes']))
                manifest['episodes'].append(record)
        (building/'collection_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        shutil.copytree(base/'collector_sources', building/'collector_sources')

        def rows(folder):
            return pd.concat([pd.read_parquet(p) for p in sorted((folder/'data').rglob('*.parquet'))], ignore_index=True).sort_values('index')

        before = rows(base)
        after = rows(building)
        retained = after[after.episode_index < len(original['episodes'])]
        for key in ('action', 'observation.state'):
            if not np.array_equal(np.stack(before[key]), np.stack(retained[key])):
                raise ValueError(f'Original data changed during merge: {key}')
        if not np.array_equal(before.timestamp.to_numpy(), retained.timestamp.to_numpy()):
            raise ValueError('Original timestamps changed')
        if len(after) != sum(e['frames'] for e in manifest['episodes']):
            raise ValueError('Merged frame count differs from manifest')
        save('validating')
        with (run/'validate200.log').open('w') as log:
            subprocess.run([sys.executable, str(ROOT/'myscript/validate_official_data.py'), '--root', str(building)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        (building/'README.md').write_text(f'# 📦 官方场景{args.total}条示范\n\n原{len(original["episodes"])}条完整保留，再补采{needed}条新种子成功示范；官方API合并，未改物理与相机。\n\n'
                                         f'原数据动作、状态和时间戳逐项一致；{args.total}条均为官方四块入盒成功回合。验证报告仅抽样解码图像并核对全部视频帧数，不代表已审看所有画面。\n\n'
                                         '来源与种子见 collection_manifest.json，检查结果见 validation_report.json。\n')
        building.rename(output)
        state['frames'] = len(after)
        state.pop('child_pid', None)
        save('completed')
    except Exception as exc:
        state['error'] = str(exc)
        save('failed')
        raise


if __name__ == '__main__':
    main()
