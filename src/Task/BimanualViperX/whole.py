import numpy as np
from dm_control import mujoco
from dataclasses import dataclass
from .base import BimanualViperXBaseTask

@dataclass
class Observation:
    qpos: np.ndarray = None
    qvel: np.ndarray = None
    env_state: np.ndarray = None
    images = dict()

class BimanualViperXTask(BimanualViperXBaseTask):

    def __init__(self):
        super().__init__()

    def initialize_robots(self, physics: mujoco.Physics, cfg):
        physics.named.data.qpos[:16] = cfg.qpos
        np.copyto(physics.data.ctrl, cfg.qpos)

    def before_step(self, action, physics):
        qposL = action[:7]
        qposR = action[7:]
        armLqpos, gLctrl = qposL[:6], self.puppet.position_unnormalize(qposL[6])
        armRqpos, gRctrl = qposR[:6], self.puppet.position_unnormalize(qposR[6])
        ctrl = np.concatenate([armLqpos, [gLctrl, -gLctrl], armRqpos, [gRctrl, -gRctrl]])
        super().before_step(ctrl, physics)  # np.copyto(physics.data.ctrl, ctrl)

    def get_observation(self, physics):
        obs = Observation()
        obs.qpos = self.get_qpos(physics)
        obs.qvel = self.get_qvel(physics)
        obs.images = dict()
        for cam_name in self.cfg.camera_names:
            obs.images[cam_name] = physics.render(height=480, width=640, camera_id=cam_name)
        return obs
