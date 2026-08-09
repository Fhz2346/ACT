import numpy as np
from dm_control import mujoco
from .Base import BaseTask
from .utils import get_contact_pairs, sample_position


class InsertionTask(BaseTask):

    def __init__(self, BVXtask, cfg):
        super().__init__(BVXtask)
        self.max_reward = 4
        self.cfg = cfg

    def randomize_initial_env(self):
        self.cfg.peg.pos.value = sample_position(self.cfg.peg.pos.ranges).tolist()
        self.cfg.soc.pos.value = sample_position(self.cfg.soc.pos.ranges).tolist()

    def initialize_episode(self, physics: mujoco.Physics):
        peg_pose = np.concatenate([self.cfg.peg.pos.value, self.cfg.peg.quat])
        soc_pose = np.concatenate([self.cfg.soc.pos.value, self.cfg.soc.quat])
        physics.data.qpos[16 : 16 + 7] = peg_pose
        physics.data.qpos[23 : 23 + 7] = soc_pose
        self.BVXtask.initialize_episode(physics)

    def get_reward(self, physics):
        contact_pairs = get_contact_pairs(physics)
        touch_right_gripper = (
            "red_peg",
            "vx300s_right/10_right_gripper_finger",
        ) in contact_pairs
        touch_left_gripper = (
            ("socket-1", "vx300s_left/10_left_gripper_finger") in contact_pairs
            or ("socket-2", "vx300s_left/10_left_gripper_finger") in contact_pairs
            or ("socket-3", "vx300s_left/10_left_gripper_finger") in contact_pairs
            or ("socket-4", "vx300s_left/10_left_gripper_finger") in contact_pairs
        )

        peg_touch_table = ("red_peg", "table") in contact_pairs
        socket_touch_table = (
            ("socket-1", "table") in contact_pairs
            or ("socket-2", "table") in contact_pairs
            or ("socket-3", "table") in contact_pairs
            or ("socket-4", "table") in contact_pairs
        )
        peg_touch_socket = (
            ("red_peg", "socket-1") in contact_pairs
            or ("red_peg", "socket-2") in contact_pairs
            or ("red_peg", "socket-3") in contact_pairs
            or ("red_peg", "socket-4") in contact_pairs
        )
        pin_touched = ("red_peg", "pin") in contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table): # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table): # peg and socket touching
            reward = 3
        if pin_touched: # successful insertion
            reward = 4
        return reward
