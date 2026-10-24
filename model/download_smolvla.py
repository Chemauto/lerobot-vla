"""Download pinned official SmolVLA assets into the ignored checkpoints directory."""

import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
BASE_REVISION = '5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6'
VLM_REVISION = '7b375e1b73b11138ff12fe22c8f2822d8fe03467'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    state = {'status': 'downloading', 'assets': []}

    def save():
        state['updated'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        temp = args.run/'assets_status.json.tmp'
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')
        temp.replace(args.run/'assets_status.json')

    try:
        save()
        vlm_repo = 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
        for repo, revision, name in [('lerobot/smolvla_base', BASE_REVISION, 'smolvla_base'),
                                     (vlm_repo, VLM_REVISION, 'smolvla_vlm')]:
            destination = ROOT/'checkpoints'/name
            item = {'repo': repo, 'revision': revision, 'path': str(destination), 'status': 'downloading'}
            state['assets'].append(item)
            save()
            snapshot_download(repo, revision=revision, local_dir=destination,
                              allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'], max_workers=4)
            item['status'] = 'completed'
            save()
        state['status'] = 'completed'
        save()
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        state.update(status='failed', error=str(exc))
        save()
        raise


if __name__ == '__main__':
    main()
