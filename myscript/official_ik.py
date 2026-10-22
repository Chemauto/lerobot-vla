"""Bounded waypoint IK, evaluated on scratch MuJoCo data (never live physics)."""
import mujoco
import numpy as np
from scipy.optimize import least_squares

class WaypointIK:
    def __init__(self, world, axis_weight=0.15):
        self.world = world
        self.model = world.model
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site('gripperframe').id
        self.body = self.model.body('gripper').id
        self.qadr = world.qadr[:5]
        self.axis_weight = axis_weight
        self.yaw = None

    def solve(self, target):
        self.data.qpos[:] = self.world.data.qpos
        current = self.world.data.qpos[self.qadr].copy()
        lo, hi = self.world.lo[:5], self.world.hi[:5]

        def residual(q):
            self.data.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.model, self.data)
            axis = self.data.xmat[self.body].reshape(3, 3)[:, 2]
            r = np.r_[self.data.site_xpos[self.site] - target,
                      self.axis_weight * (axis - [0, 0, 1])]
            if self.yaw is not None:
                x_axis = self.data.xmat[self.body].reshape(3, 3)[:, 0]
                r = np.r_[r, .1 * (x_axis - [np.cos(self.yaw), np.sin(self.yaw), 0])]
            return r

        candidates = []
        for start in (current, np.zeros(5)):
            result = least_squares(residual, np.clip(start, lo+1e-6, hi-1e-6),
                                   bounds=(lo, hi), max_nfev=80, gtol=1e-7)
            r = residual(result.x)
            candidates.append((np.linalg.norm(r), result.x, float(np.linalg.norm(r[:3]))))
            if np.linalg.norm(r) < 0.003:
                break
        _, q, error = min(candidates, key=lambda c: c[0])
        return q, error
