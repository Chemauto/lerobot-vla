"""Collect successful official-tabletop expert episodes as a local LeRobot dataset.

Frames are (observation BEFORE action, action actually executed). No training
or publishing is triggered. Existing output directories are never overwritten.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo

import av
import numpy as np
av.logging.set_level(av.logging.ERROR)
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from official_env import make_official_env, REPO, REVISION
from official_teacher import OfficialTeacher, INITIAL_JOINTS
from media_utils import save_mp4

ROOT = Path(__file__).resolve().parents[1]
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=5, help='Number of successful episodes to retain')
    parser.add_argument('--max-attempts', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--root', type=Path, default=None)
    parser.add_argument('--repo-id', default='local/so101_official_tabletop')
    parser.add_argument('--max-passes', type=int, default=3)
    parser.add_argument('--preview', action='store_true', help='Save first successful rollout as a side-by-side video')
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'configs/official_tabletop.json')
    args = parser.parse_args()
    if args.root is None:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
        args.root = ROOT / 'data' / f'{stamp}_官方场景示范'
    if args.root.exists():
        parser.error(f'Output exists: {args.root}. Choose a new --root; no files were replaced.')
    if args.episodes < 1 or args.max_attempts < args.episodes:
        parser.error('Require 1 <= episodes <= max-attempts')
    settings = json.loads(args.config.read_text())
    variant = settings.get('task_variant')
    target_count = variant['cube_count'] if variant else 4
    envs = make_official_env(config_path=args.config, initial_joints=INITIAL_JOINTS)
    raw = envs['so101_tabletop'][0].envs[0]
    features = {
        'observation.images.front': {'dtype': 'video', 'shape': (480, 640, 3), 'names': ['height', 'width', 'channels']},
        'observation.images.wrist': {'dtype': 'video', 'shape': (480, 640, 3), 'names': ['height', 'width', 'channels']},
        'observation.state': {'dtype': 'float32', 'shape': (6,), 'names': JOINTS},
        'action': {'dtype': 'float32', 'shape': (6,), 'names': JOINTS},
    }
    ds = LeRobotDataset.create(
        args.repo_id, root=args.root, fps=raw.fps, features=features,
        robot_type='so101_follower', use_videos=True, video_backend='pyav',
        streaming_encoding=True, encoder_queue_maxsize=16, encoder_threads=2,
        rgb_encoder=RGBEncoderConfig(vcodec='h264', crf=23, preset='fast', g=2),
        metadata_buffer_size=1,
    )
    sources = args.root/'collector_sources'
    sources.mkdir()
    hashes = {}
    source_names = ['official_teacher.py', 'official_ik.py', 'official_env.py', 'collect_official.py', 'media_utils.py']
    if variant: source_names.append('task_variants.py')
    for name in source_names:
        path = Path(__file__).parent/name
        shutil.copy2(path, sources/name)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (sources/'configs').mkdir()
    shutil.copy2(args.config, sources/'configs/official_tabletop.json')
    versions = {name: importlib.metadata.version(name) for name in ['lerobot', 'mujoco', 'numpy', 'scipy', 'av']}
    versions['python'] = platform.python_version()
    manifest = dict(versions=versions, env_repo=REPO, env_revision=REVISION, fps=raw.fps,
                    initial_joints=INITIAL_JOINTS, randomization=settings.get('randomize', 'official defaults'),
                    physics_overrides=False, sources_sha256=hashes,
                    state_units='arm: degrees; gripper: normalized 0..100',
                    alignment='observation before action', episodes=[], attempts=[])
    if variant:
        manifest.update(task_variant=variant, target_cube_count=target_count,
            derived_task=True, strict_success='All target cubes in box; action and actual gripper>=95 for final66 consecutive frames',
            provenance='Pinned official environment, locally configured task constants and external fixed/single-cube scene; contact physics unchanged')
    kept = 0
    preview_saved = False
    try:
        for attempt in range(args.max_attempts):
            seed = args.seed+attempt
            raw.reset(seed=seed)
            initial_qpos = raw._world.data.qpos.copy().tolist()
            phase_changes = []
            frame_number = 0
            last_phase = None
            stable_frames = 0
            instruction = raw.task_description

            def record(obs, action, phase):
                nonlocal frame_number, last_phase, stable_frames
                stable = raw._world.cubes_in_box() == target_count and obs['agent_pos'][5] >= 95 and action[5] >= 95
                stable_frames = stable_frames+1 if stable else 0
                if phase != last_phase:
                    phase_changes.append(dict(frame=frame_number, phase=phase))
                    last_phase = phase
                ds.add_frame({
                    'observation.images.front': obs['pixels']['cam0'],
                    'observation.images.wrist': obs['pixels']['cam1'],
                    'observation.state': obs['agent_pos'].astype(np.float32),
                    'action': action.astype(np.float32), 'task': instruction,
                })
                frame_number += 1

            teacher = OfficialTeacher(raw, video=args.preview and not preview_saved,
                                      max_passes=args.max_passes, on_frame=record)
            result = teacher.run()
            if variant and result['success']:
                teacher.phase = 'stable_finish'
                teacher.hold(100, 2.2)
                result.update(cubes_in_box=raw._world.cubes_in_box(), steps=teacher.steps,
                    truncated=teacher.done, final_stable_frames=stable_frames,
                    success=bool(raw._world.cubes_in_box() == target_count and stable_frames >= 66 and not teacher.done))
            manifest['attempts'].append(dict(seed=seed, success=result['success'],
                                              cubes_in_box=result['cubes_in_box'], frames=frame_number))
            if result['success']:
                ds.save_episode(parallel_encoding=False)
                manifest['episodes'].append(dict(episode_index=kept, seed=seed,
                    frames=frame_number, initial_qpos=initial_qpos,
                    final_qpos=raw._world.data.qpos.copy().tolist(),
                    final_state=raw._world.joint_state().tolist(),
                    phases=phase_changes, result=result))
                kept += 1
                if teacher.frames:
                    save_mp4(teacher.frames, str(args.root/'preview_success.mp4'), fps=raw.fps)
                    preview_saved = True
                print(f'KEPT {kept}/{args.episodes}: seed={seed}, frames={frame_number}', flush=True)
            else:
                ds.clear_episode_buffer()
                print(f'DISCARDED seed={seed}: {result["cubes_in_box"]}/{target_count} cubes', flush=True)
            (args.root/'collection_manifest.json').write_text(json.dumps(manifest, indent=2))
            if kept >= args.episodes:
                break
    finally:
        ds.finalize()
        raw.close()
    print(json.dumps(dict(saved=kept, requested=args.episodes, root=str(args.root)), indent=2), flush=True)
    if kept < args.episodes:
        raise SystemExit('Attempt cap reached; completed episodes remain saved. Increase cap for a new run.')

if __name__ == '__main__':
    main()
