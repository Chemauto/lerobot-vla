"""Download the pinned official scene into the external Hugging Face cache."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
from official_env import REPO, REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Verify an already downloaded snapshot')
    args = parser.parse_args()
    path = Path(snapshot_download(
        REPO, revision=REVISION, local_files_only=args.offline,
        allow_patterns=['env.py', 'requirements.txt', 'so101_tabletop.xml',
                        'robotstudio_so101/**', 'textures/**'],
    ))
    for required in ['env.py', 'so101_tabletop.xml', 'robotstudio_so101/so101.xml', 'textures/wood.png']:
        if not (path/required).is_file():
            raise FileNotFoundError(f'Missing official asset: {path/required}')
    print(f'Official assets ready: {path}')

if __name__ == '__main__':
    main()
