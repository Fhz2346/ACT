import numpy as np
from dataclasses import dataclass


@dataclass
class Waypoint:
    t: int  # timestep
    xyz: np.ndarray  # position
    quat: np.ndarray  # quaternion
    gripper: float  # 0: close, 1: open


def interpolate(cur: Waypoint, nxt: Waypoint, t):
    t_frac = (t - cur.t) / (nxt.t - cur.t)
    xyz = cur.xyz + (nxt.xyz - cur.xyz) * t_frac
    quat = cur.quat + (nxt.quat - cur.quat) * t_frac
    gripper = cur.gripper + (nxt.gripper - cur.gripper) * t_frac
    return Waypoint(t, xyz, quat, gripper)
