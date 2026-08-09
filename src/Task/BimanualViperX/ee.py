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
    mocapLpose: np.ndarray = None
    mocapRpose: np.ndarray = None
    gctrl: np.ndarray = None


class BimanualViperXEETask(BimanualViperXBaseTask):

    def __init__(self):
        super().__init__()

    def initialize_robots(self, physics: mujoco.Physics, cfg):
        physics.named.data.qpos[:16] = cfg.qpos

        def shift_mocap_pos(cfg):
            posL = cfg.left.pos
            posL[0] += 0.1
            posR = cfg.right.pos
            posR[0] -= 0.1
            return posL, posR

        posL, posR = shift_mocap_pos(cfg.mocap)
        np.copyto(physics.data.mocap_pos[0], posL)
        np.copyto(physics.data.mocap_quat[0], cfg.mocap.left.quat)
        np.copyto(physics.data.mocap_pos[1], posR)
        np.copyto(physics.data.mocap_quat[1], cfg.mocap.right.quat)
        PC = self.puppet.POSITION_CLOSE
        np.copyto(physics.data.ctrl, [PC, -PC, PC, -PC])

    def parse_action_side(self, action):
        # action: [pos(3), quat(4), gctrl(1)]
        assert len(action) == 8
        pos, quat, gctrl_n = action[:3], action[3:7], action[7]
        gctrl = self.puppet.position_unnormalize(gctrl_n)
        return pos, quat, gctrl

    def before_step(self, action, physics: mujoco.Physics):
        assert len(action) == 16
        posL, quatL, gLctrl = self.parse_action_side(action[:8])
        posR, quatR, gRctrl = self.parse_action_side(action[8:])
        np.copyto(physics.data.mocap_pos[0], posL)
        np.copyto(physics.data.mocap_quat[0], quatL)
        np.copyto(physics.data.mocap_pos[1], posR)
        np.copyto(physics.data.mocap_quat[1], quatR)
        gctrl = [gLctrl, -gLctrl, gRctrl, -gRctrl]
        np.copyto(physics.data.ctrl, gctrl)

    def get_observation(self, physics):
        obs = Observation()
        obs.qpos = self.get_qpos(physics)
        obs.qvel = self.get_qvel(physics)
        obs.images = dict()
        obs.images["top"] = physics.render(height=480, width=640, camera_id="top")
        # obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        # obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        obs.mocapLpose = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        obs.mocapRpose = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()
        obs.gctrl = physics.data.ctrl.copy() # used when replaying joint trajectory
        return obs

    def get_reward(self, physics):
        raise NotImplementedError
