import numpy as np
from dm_control import mujoco
from dm_control.suite import base
from omegaconf import OmegaConf

from .Gripper import Gripper
from Pathes import P


class BimanualViperXBaseTask(base.Task):

    def __init__(self, cfg=OmegaConf.load(P.config_dir / "BimanualViperX.yaml")):
        super().__init__(random=False)
        self.cfg = cfg
        self.master = Gripper(cfg.grippers.master)
        self.puppet = Gripper(cfg.grippers.puppet)

    def initialize_robots(self, physics: mujoco.Physics, cfg):
        raise NotImplementedError

    def initialize_episode(self, physics):
        self.initialize_robots(physics, self.cfg.initial)
        super().initialize_episode(physics)

    def get_qpos(self, physics):
        qpos_raw = physics.data.qpos.copy()
        qposLraw = qpos_raw[:8]
        qposRraw = qpos_raw[8:16]
        armLqpos, gLqpos = qposLraw[:6], [self.puppet.position_normalize(qposLraw[6])]
        armRqpos, gRqpos = qposRraw[:6], [self.puppet.position_normalize(qposRraw[6])]
        qpos = np.concatenate([armLqpos, gLqpos, armRqpos, gRqpos])
        return qpos

    def get_qvel(self, physics):
        qvel_raw = physics.data.qvel.copy()
        qvelLraw = qvel_raw[:8]
        qvelRraw = qvel_raw[8:16]
        armLqvel, gLqvel = qvelLraw[:6], [self.puppet.velocity_normalize(qvelLraw[6])]
        armRqvel, gRqvel = qvelRraw[:6], [self.puppet.velocity_normalize(qvelRraw[6])]
        return np.concatenate([armLqvel, gLqvel, armRqvel, gRqvel])

    def get_env_state(self, physics):
        raise NotImplementedError

    def get_reward(self, physics):
        raise NotImplementedError
