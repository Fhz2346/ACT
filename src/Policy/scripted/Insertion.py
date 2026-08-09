import numpy as np
from pyquaternion import Quaternion
from .Base import BasePolicy, Waypoint
from Task.BimanualViperX.ee import Observation

class InsertionPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        obs: Observation = ts_first.observation
        initRmocap_pose = obs.mocapRpose
        initLmocap_pose = obs.mocapLpose

        peg_info = np.array(obs.env_state)[:7]
        peg_xyz = peg_info[:3]
        peg_quat = peg_info[3:]

        socket_info = np.array(obs.env_state)[7:]
        socket_xyz = socket_info[:3]
        socket_quat = socket_info[3:]

        gripperRpick_quat = Quaternion(initRmocap_pose[3:]) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)
        gripperLpick_quat = Quaternion(initRmocap_pose[3:]) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=+60)

        meet_xyz = np.array([0, 0.5, 0.15])
        lift_right = 0.00715

        self.trajectoryL = [
            Waypoint(0, initLmocap_pose[:3], initLmocap_pose[3:], 0),                                 # sleep
            Waypoint(120, socket_xyz + np.array([0, 0, 0.08]), gripperLpick_quat.elements, 1),        # approach the cube
            Waypoint(170, socket_xyz + np.array([0, 0, -0.03]), gripperLpick_quat.elements, 1),       # go down
            Waypoint(220, socket_xyz + np.array([0, 0, -0.03]), gripperLpick_quat.elements, 0),       # close gripper
            Waypoint(285, meet_xyz + np.array([-0.1, 0, 0]), gripperLpick_quat.elements, 0),          # approach meet position
            Waypoint(340, meet_xyz + np.array([-0.05, 0, 0]), gripperLpick_quat.elements, 0),         # insertion
            Waypoint(400, meet_xyz + np.array([-0.05, 0, 0]), gripperLpick_quat.elements, 0),         # insertion
        ]
        self.trajectoryR = [
            Waypoint(0, initRmocap_pose[:3], initRmocap_pose[3:], 0),                                 # sleep
            Waypoint(120, peg_xyz + np.array([0, 0, 0.08]), gripperRpick_quat.elements, 1),           # approach the cube
            Waypoint(170, peg_xyz + np.array([0, 0, -0.03]), gripperRpick_quat.elements, 1),          # go down
            Waypoint(220, peg_xyz + np.array([0, 0, -0.03]), gripperRpick_quat.elements, 0),          # close gripper
            Waypoint(285, meet_xyz + np.array([0.1, 0, lift_right]), gripperRpick_quat.elements, 0),  # approach meet position
            Waypoint(340, meet_xyz + np.array([0.05, 0, lift_right]), gripperRpick_quat.elements, 0), # insertion
            Waypoint(400, meet_xyz + np.array([0.05, 0, lift_right]), gripperRpick_quat.elements, 0), # insertion
        ]
