"""Scripted expert for the pinned official SO-101 tabletop environment.

Read privileged cube poses -> solve downward-facing waypoint IK -> execute
smooth joint targets -> verify physical lift and placement -> retry failures.
Upstream sources, contact parameters and object state are never modified.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from official_env import make_official_env
from official_ik import WaypointIK
from media_utils import save_mp4

ROOT = Path(__file__).resolve().parents[1]
INITIAL_JOINTS = [0, -90, 90, 37.815214, 0, 0]

class OfficialTeacher:
    def __init__(self, raw, video=False, axis_weight=.15, controller='ik',
                 max_passes=3, on_frame=None):
        self.env = raw
        self.w = raw._world
        self.m, self.d = self.w.model, self.w.data
        self.controller = controller
        self.ik = WaypointIK(self.w, axis_weight)
        self.axis_weight = axis_weight
        self.gripper_body = self.m.body('gripper').id
        self.site = self.m.site('gripperframe').id
        self.dofs = np.array([self.m.joint(j).dofadr[0] for j in
                              ('shoulder_pan', 'shoulder_lift', 'elbow_flex',
                               'wrist_flex', 'wrist_roll')])
        self.jp = np.zeros((3, self.m.nv))
        self.jr = np.zeros_like(self.jp)
        self.frames, self.video = [], video
        self.info, self.done, self.steps = {}, False, 0
        self.phase = 'start'
        self.max_passes = max_passes
        self.on_frame = on_frame
        self.obs = self.env._obs(self.w)
        self.terminal_seen = False

    def execute(self, q, grip):
        action = np.r_[np.rad2deg(q[:5]), grip].astype(np.float32)
        action[:5] = np.clip(action[:5], np.rad2deg(self.w.lo[:5]),
                            np.rad2deg(self.w.hi[:5]))
        action[5] = np.clip(action[5], 0, 100)
        if self.on_frame:
            self.on_frame(self.obs, action.copy(), self.phase)
        self.obs, _, term, trunc, self.info = self.env.step(action)
        self.steps += 1
        self.terminal_seen |= bool(term)
        # A raw Gymnasium env permits physics stepping after success. Complete
        # release/retreat and check settled success, rather than accept a held cube.
        self.done = bool(trunc)
        if self.video:
            self.frames.append(np.concatenate([self.obs['pixels']['cam0'],
                                               self.obs['pixels']['cam1']], axis=1))

    def go(self, target, grip, seconds, speed=1.2):
        if self.done:
            return
        target = np.asarray(target)
        if self.controller == 'ik':
            start = self.d.qpos[self.w.qadr[:5]].copy()
            goal, _ = self.ik.solve(target)
            duration = max(seconds, 1.5 * np.max(np.abs(goal-start)) / speed)
            count = max(1, round(duration*self.env.fps))
            for i in range(count):
                if self.done:
                    return
                t = (i+1)/count
                t = t*t*(3-2*t)
                self.execute(np.r_[start+t*(goal-start), 0.], grip)
            return
        for _ in range(max(1, round(seconds*self.env.fps))):
            if self.done:
                return
            mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.site)
            j = self.jp[:, self.dofs]
            error = target-self.d.site_xpos[self.site]
            axis = self.d.xmat[self.gripper_body].reshape(3, 3)[:, 2]
            skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                             [-axis[1], axis[0], 0]])
            j = np.vstack([j, -self.axis_weight*skew@self.jr[:, self.dofs]])
            error = np.r_[error, self.axis_weight*(np.array([0, 0, 1])-axis)]
            dq = .6*j.T@np.linalg.solve(j@j.T+.025**2*np.eye(6), error)
            q = self.d.qpos[self.w.qadr].copy()
            q[:5] += np.clip(dq, -speed/self.env.fps, speed/self.env.fps)
            self.execute(q, grip)

    def hold(self, grip, seconds):
        q = self.d.qpos[self.w.qadr].copy()
        for _ in range(max(1, round(seconds*self.env.fps))):
            if self.done:
                return
            self.execute(q, grip)

    def inside(self, body):
        local = (self.d.xpos[body]-self.d.xpos[self.w.box_body]) @ self.d.xmat[self.w.box_body].reshape(3, 3)
        return bool(np.all(np.abs(local[:2]) < .045) and local[2] < .06)

    def drop_slot(self, body):
        box = self.d.xpos[self.w.box_body]
        rot = self.d.xmat[self.w.box_body].reshape(3, 3)
        occupied = [self.d.xpos[b][:2] for b in self.w.cube_bodies
                    if b != body and self.inside(b)]
        slots = [box + rot @ np.array([x, y, 0.])
                 for x, y in ((-.02, -.02), (.02, -.02), (-.02, .02), (.02, .02))]
        if not occupied:
            return min(slots, key=lambda s: np.linalg.norm(s[:2]))[:2]
        return max(slots, key=lambda s: min(np.linalg.norm(s[:2]-p) for p in occupied))[:2]

    def pick_place(self, cube_index):
        body = self.w.cube_bodies[cube_index]
        p = self.d.xpos[body].copy()
        cube_rotation = self.d.xmat[body].reshape(3, 3)
        cube_yaw = np.arctan2(cube_rotation[1, 0], cube_rotation[0, 0])
        gripper_rotation = self.d.xmat[self.gripper_body].reshape(3, 3)
        gripper_yaw = np.arctan2(gripper_rotation[1, 0], gripper_rotation[0, 0])
        # Square cubes admit four equivalent face-aligned grasp directions.
        self.ik.yaw = cube_yaw + np.round((gripper_yaw-cube_yaw)/(np.pi/2))*(np.pi/2)
        self.phase = 'approach'
        self.go([p[0], p[1], .10], 100, 1.0)
        # Refresh the privileged position after approach (accidental contact can move it).
        p = self.d.xpos[body].copy()
        self.phase = 'descend'
        self.go([p[0], p[1], .011], 100, 1.3, speed=.75)
        self.phase = 'close'
        self.hold(0, .7)
        self.phase = 'lift'
        self.go([p[0], p[1], .10], 0, 1.2, speed=.8)
        lifted = float(self.d.xpos[body, 2]-p[2])
        if lifted > .035 and not self.done:
            target = self.drop_slot(body)
            offset = np.clip(self.d.xpos[body, :2]-self.d.site_xpos[self.site, :2], -.025, .025)
            target_site = target-offset
            self.phase = 'carry'
            self.go([*target_site, .10], 0, 1.5, speed=.8)
            self.phase = 'lower'
            self.go([*target_site, .045], 0, 1.0, speed=.6)
            self.phase = 'release'
            for grip in np.linspace(0, 100, 25):
                self.hold(float(grip), 1/self.env.fps)
            self.hold(100, .5)
            self.phase = 'retreat'
            self.go([*target_site, .12], 100, .9, speed=.9)
        else:
            self.phase = 'recover'
            self.hold(100, .3)
            self.go([p[0], p[1], .12], 100, .6)
        return dict(cube=cube_index, lift=lifted, placed=self.inside(body),
                    cubes_in_box=self.w.cubes_in_box())

    def run(self):
        records = []
        target_count = len(self.w.cube_bodies)
        for attempt in range(self.max_passes):
            for i, body in enumerate(self.w.cube_bodies):
                if self.done:
                    break
                if self.inside(body):
                    continue
                result = self.pick_place(i)
                result['pass'] = attempt+1
                records.append(result)
                print(result, flush=True)
            if self.w.cubes_in_box() == target_count or self.done:
                break
        self.phase = 'settle'
        self.hold(100, .8)
        count = self.w.cubes_in_box()
        # Stronger than raw termination: gripper is open and all four remain inside.
        success = count == target_count and self.info.get('is_success', False) and not self.done
        return dict(records=records, cubes_in_box=count, success=bool(success),
                    terminal_seen=self.terminal_seen, steps=self.steps,
                    truncated=self.done)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--out-dir', type=Path, default=ROOT/'.cache/teacher_demos')
    parser.add_argument('--controller', choices=['ik', 'dls'], default='ik')
    parser.add_argument('--max-passes', type=int, default=3)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    envs = make_official_env(initial_joints=INITIAL_JOINTS)
    raw = envs['so101_tabletop'][0].envs[0]
    results = []
    try:
        for ep in range(args.episodes):
            raw.reset(seed=args.seed+ep)
            teacher = OfficialTeacher(raw, args.video and ep == 0,
                                      controller=args.controller, max_passes=args.max_passes)
            result = teacher.run()
            result['seed'] = args.seed+ep
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)
            if teacher.frames:
                save_mp4(teacher.frames, str(args.out_dir/f'seed{args.seed+ep}.mp4'), fps=raw.fps)
            (args.out_dir/'results.json').write_text(json.dumps(dict(
                official_repo='lerobot/so101-tabletop-mujoco', controller=args.controller,
                initial_joints=INITIAL_JOINTS, fps=raw.fps, episodes=results), indent=2))
    finally:
        raw.close()

if __name__ == '__main__':
    main()
