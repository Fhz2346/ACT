import numpy as np
from .Base import BaseTask
from .utils import get_contact_pairs, sample_position

class TransferCubeTask(BaseTask):

    def __init__(self, BVXtask, cfg):
        super().__init__(BVXtask)
        self.max_reward = 4
        self.cfg = cfg

    def randomize_initial_env(self):
        self.cfg.box.pos.value = sample_position(self.cfg.box.pos.ranges).tolist()

    def initialize_episode(self, physics):
        box_pose = np.concatenate([self.cfg.box.pos.value, self.cfg.box.quat])
        physics.named.data.qpos[-7:] = box_pose
        self.BVXtask.initialize_episode(physics)

    def get_reward(self, physics):
        contact_pairs = get_contact_pairs(physics)
        is_box_touch_Lgripper = ("red_box", "vx300s_left/10_left_gripper_finger") in contact_pairs
        is_box_touch_Rgripper = ("red_box", "vx300s_right/10_right_gripper_finger") in contact_pairs
        is_box_touch_table = ("red_box", "table") in contact_pairs

        reward = 0
        if is_box_touch_Rgripper:
            reward = 1 
        if is_box_touch_Rgripper and not is_box_touch_table: 
            reward = 2 # lifted
        if is_box_touch_Lgripper: 
            reward = 3 # attempted transfer
        if is_box_touch_Lgripper and not is_box_touch_table: 
            reward = 4 # successful transfer
        return reward
