import numpy as np
from pyquaternion import Quaternion
from .Base import BasePolicy, Waypoint
from Task.BimanualViperX.ee import Observation


class PickAndTransferPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        obs: Observation = ts_first.observation
        init_mocapLpose = obs.mocapLpose
        init_mocapRpose = obs.mocapRpose
        box_info = obs.env_state
        box_xyz, box_quat = box_info[:3], box_info[3:]
        gripper_pick_quat = Quaternion(init_mocapRpose[3:]) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)
        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)
        meet_xyz = np.array([0, 0.5, 0.25])

        self.trajectoryL = [
            Waypoint(  0, init_mocapLpose[:3], init_mocapLpose[3:], 0),                         # sleep
            Waypoint(100, meet_xyz + np.array([-0.1, 0, -0.02]), meet_left_quat.elements, 1),   # approach meet position
            Waypoint(260, meet_xyz + np.array([0.02, 0, -0.02]), meet_left_quat.elements, 1),   # move to meet position
            Waypoint(310, meet_xyz + np.array([0.02, 0, -0.02]), meet_left_quat.elements, 0),   # close gripper
            Waypoint(360, meet_xyz + np.array([-0.1, 0, -0.02]), np.array([1, 0, 0, 0]), 0),    # move left
            Waypoint(400, meet_xyz + np.array([-0.1, 0, -0.02]), np.array([1, 0, 0, 0]), 0),    # stay
        ]
        self.trajectoryR = [
            Waypoint(  0, init_mocapRpose[:3], init_mocapRpose[3:], 0),                         # initial pose
            Waypoint( 90, box_xyz + np.array([0, 0, 0.08]), gripper_pick_quat.elements, 1),     # approach the cube
            Waypoint(130, box_xyz + np.array([0, 0, -0.015]), gripper_pick_quat.elements, 1),   # go down
            Waypoint(170, box_xyz + np.array([0, 0, -0.015]), gripper_pick_quat.elements, 0),   # close gripper
            Waypoint(200, meet_xyz + np.array([0.05, 0, 0]), gripper_pick_quat.elements, 0),    # approach meet position
            Waypoint(220, meet_xyz, gripper_pick_quat.elements, 0),                             # move to meet position
            Waypoint(310, meet_xyz, gripper_pick_quat.elements, 1),                             # open gripper
            Waypoint(360, meet_xyz + np.array([0.1, 0, 0]), gripper_pick_quat.elements, 1),     # move to right
            Waypoint(400, meet_xyz + np.array([0.1, 0, 0]), gripper_pick_quat.elements, 1),     # stay
        ]
