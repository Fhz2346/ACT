import numpy as np
from dm_control import mujoco

def sample_position(ranges):
    # x_range, y_range, z_range = ranges
    ranges = np.vstack(ranges)
    # np.random.seed(42)
    position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    return position


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])
    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])
    return peg_pose, socket_pose


def get_contact_pairs(physics: mujoco.Physics):
    contact_pairs = []
    for i in range(physics.data.ncon):
        geom_1_id = physics.data.contact[i].geom1
        geom_2_id = physics.data.contact[i].geom2
        geom_1_name = physics.model.id2name(geom_1_id, "geom")
        geom_2_name = physics.model.id2name(geom_2_id, "geom")
        contact_pair = (geom_1_name, geom_2_name)
        contact_pairs.append(contact_pair)
    return contact_pairs
