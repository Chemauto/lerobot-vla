"""Check local expert data: successes, alignment, timestamps, chunks and video decode."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
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
    parser.add_argument('--replay-physics', action='store_true', help='Exactly replay full-reset task-variant episodes and verify stable release')
    args = parser.parse_args()
    manifest = json.loads((args.root/'collection_manifest.json').read_text())
    fps = manifest['fps']
    target_count = manifest.get('target_cube_count', 4)
    dataset = LeRobotDataset('local/so101_official_tabletop', root=args.root,
        video_backend='pyav', delta_timestamps={'action': [i/fps for i in range(10)]})
    rows = pd.concat([pd.read_parquet(p) for p in sorted((args.root/'data').rglob('*.parquet'))]).sort_values(['episode_index','frame_index'])
    expected = sum(ep['frames'] for ep in manifest['episodes'])
    assert len(rows) == expected == dataset.meta.total_frames
    assert dataset.meta.total_episodes == len(manifest['episodes']) > 0
    actions = np.stack(rows['action'])
    states = np.stack(rows['observation.state'])
    assert np.isfinite(actions).all() and np.isfinite(states).all()
    assert np.all((actions[:, 5] >= 0) & (actions[:, 5] <= 100))
    for ep in manifest['episodes']:
        assert ep['result']['success'] and ep['result']['cubes_in_box'] == target_count
        if manifest.get('task_variant'):
            assert ep['result']['final_stable_frames'] >= 66
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
    replay_report = None
    if args.replay_physics:
        if not manifest.get('task_variant'):
            parser.error('--replay-physics requires full-reset task variants, not mid-episode correction segments')
        import mujoco
        from official_env import make_official_env
        env = make_official_env(config_path=args.root/'collector_sources/configs/official_tabletop.json',
            initial_joints=manifest['initial_joints'])['so101_tabletop'][0].envs[0]
        checks=[]
        try:
            for episode in manifest['episodes']:
                env.reset(seed=episode['seed']);w=env._world
                state_error=0.;stable=0
                part=rows[rows.episode_index==episode['episode_index']]
                for action,state in zip(part['action'],part['observation.state']):
                    state_error=max(state_error,float(np.abs(w.joint_state()-state).max()))
                    good=w.cubes_in_box()==target_count and w.joint_state()[5]>=95 and action[5]>=95
                    stable=stable+1 if good else 0
                    w.data.ctrl[w.actuators]=w.to_ctrl(np.asarray(action,dtype=np.float32))
                    mujoco.mj_step(w.model,w.data,nstep=w.substeps);env._steps+=1
                final_error=float(np.abs(w.data.qpos-np.asarray(episode['final_qpos'])).max())
                assert state_error<1e-5 and final_error<1e-9
                assert w.cubes_in_box()==target_count and stable>=66
                checks.append(dict(episode_index=episode['episode_index'],seed=episode['seed'],
                    state_max_error=state_error,final_qpos_max_error=final_error,final_stable_frames=stable))
                print(f'REPLAY {len(checks)}/{len(manifest["episodes"])}: seed={episode["seed"]}, state_error={state_error}',flush=True)
        finally:env.close()
        replay_report=dict(episodes=len(checks),checks=checks)
    report = dict(episodes=dataset.meta.total_episodes, frames=expected, fps=fps,
                  target_cube_count=target_count,
                  successful_only=True, finite=True, before_action_alignment=True,
                  action_chunk_shape=[10, 6], full_video_decode=args.decode_all, videos=videos)
    if replay_report is not None:report['physics_replay']=replay_report
    (args.root/'validation_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
