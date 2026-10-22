"""Check local expert data: successes, alignment, timestamps, chunks and video decode."""
import argparse
import json
from pathlib import Path
import av
import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--decode-all', action='store_true', help='Decode every video frame, not just sampled training frames')
    args = parser.parse_args()
    manifest = json.loads((args.root/'collection_manifest.json').read_text())
    fps = manifest['fps']
    dataset = LeRobotDataset('local/so101_official_tabletop', root=args.root,
        video_backend='pyav', delta_timestamps={'action': [i/fps for i in range(10)]})
    rows = pd.concat([pd.read_parquet(p) for p in (args.root/'data').rglob('*.parquet')])
    expected = sum(ep['frames'] for ep in manifest['episodes'])
    assert len(rows) == expected == dataset.meta.total_frames
    assert dataset.meta.total_episodes == len(manifest['episodes']) > 0
    actions = np.stack(rows['action'])
    states = np.stack(rows['observation.state'])
    assert np.isfinite(actions).all() and np.isfinite(states).all()
    assert np.all((actions[:, 5] >= 0) & (actions[:, 5] <= 100))
    for ep in manifest['episodes']:
        assert ep['result']['success'] and ep['result']['cubes_in_box'] == 4
        assert ep['result']['terminal_seen'] and not ep['result']['truncated']
        part = rows[rows.episode_index == ep['episode_index']]
        assert len(part) == ep['frames']
        assert np.allclose(part.timestamp, np.arange(len(part))/fps, atol=1e-4)
        # This pinned scene declares the six arm joints after the cube free joints.
        first = np.rad2deg(np.asarray(ep['initial_qpos'])[-6:])
        first[5] = (first[5]+10)/110*100
        assert np.allclose(part.iloc[0]['observation.state'], first, atol=.01)
        assert abs(part.iloc[-1]['action'][5]-100) < .001
        assert abs(part.iloc[-1]['observation.state'][5]-100) < 1
    for index in np.linspace(0, len(dataset)-1, min(12, len(dataset)), dtype=int):
        sample = dataset[int(index)]
        assert tuple(sample['action'].shape) == (10, 6)
        for key in ['action', 'observation.state', 'observation.images.front', 'observation.images.wrist']:
            assert sample[key].isfinite().all()
        assert tuple(sample['observation.images.front'].shape) == (3, 480, 640)
    videos = []
    for path in sorted((args.root/'videos').rglob('*.mp4')):
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            count = sum(1 for _ in container.decode(video=0)) if args.decode_all else stream.frames
            videos.append(dict(file=str(path.relative_to(args.root)), frames=count, fps=str(stream.average_rate)))
            assert float(stream.average_rate) == fps
    assert sum(v['frames'] for v in videos) == 2*expected
    report = dict(episodes=dataset.meta.total_episodes, frames=expected, fps=fps,
                  successful_only=True, finite=True, before_action_alignment=True,
                  action_chunk_shape=[10, 6], full_video_decode=args.decode_all, videos=videos)
    (args.root/'validation_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
