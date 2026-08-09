import numpy as np
from dataclasses import dataclass
from .utils import Waypoint, interpolate


@dataclass
class WaypointBundle:
    curL: Waypoint = None
    nxtL: Waypoint = None
    curR: Waypoint = None
    nxtR: Waypoint = None


class BasePolicy:

    def __init__(self):
        self.t = 0
        self.wpb = WaypointBundle()

    def generate_trajectory(self, ts_first):
        self.trajectoryL = None
        self.trajectoryR = None
        raise NotImplementedError

    def update_waypoint_bundle(self):
        if self.trajectoryL[0].t == self.t:
            self.wpb.curL = self.trajectoryL.pop(0)
        self.wpb.nxtL = self.trajectoryL[0]

        if self.trajectoryR[0].t == self.t:
            self.wpb.curR = self.trajectoryR.pop(0)
        self.wpb.nxtR = self.trajectoryR[0]

    def __call__(self, ts):
        self.update_waypoint_bundle()
        wpL = interpolate(self.wpb.curL, self.wpb.nxtL, self.t)
        wpR = interpolate(self.wpb.curR, self.wpb.nxtR, self.t)
        actionL = np.concatenate([wpL.xyz, wpL.quat, [wpL.gripper]])
        actionR = np.concatenate([wpR.xyz, wpR.quat, [wpR.gripper]])
        action = np.concatenate([actionL, actionR])
        self.t += 1
        return action


class BasePolicyNoisy(BasePolicy):

    def __init__(self, noise_scale=0.01):
        super().__init__()
        self.noise_scale = noise_scale

    def __call__(self, ts):
        action = super().__call__(ts)
        wpL_xyz = action[:3]
        wpR_xyz = action[8:11]
        wpL_xyz = wpL_xyz + np.random.uniform(-self.noise_scale, self.noise_scale, wpL_xyz.shape)
        wpR_xyz = wpR_xyz + np.random.uniform(-self.noise_scale, self.noise_scale, wpR_xyz.shape)
        action[:3] = wpL_xyz
        action[8:11] = wpR_xyz
        return action
