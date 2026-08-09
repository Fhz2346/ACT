import numpy as np
from omegaconf import OmegaConf
import h5py
import os

from Policy.scripted import task_name2policy
from Pathes import P

cfg = OmegaConf.load(P.config_dir / "demonstration.yaml")
cfg.task = OmegaConf.load(P.config_dir / f"Task/{cfg.task.name}.yaml")
cfg.robot = OmegaConf.load(P.config_dir / f"BimanualViperX.yaml")

def evaluate_episode(env, episode):
    rewards = [ts.reward for ts in episode[1:]]
    episode_max_reward = np.max(rewards)
    episode_return = np.sum(rewards)
    success = episode_max_reward == env.task.max_reward
    return success, episode_return


def rollout_ee(env):
    ts = env.reset()
    episode = [ts]
    policy = task_name2policy[cfg.task.name]()
    policy.generate_trajectory(ts) # generate trajectory at first timestep, then open-loop execution
    for t in range(cfg.episode_len - 1):
        action = policy(ts)
        ts = env.step(action)
        episode.append(ts)
    return episode


def get_normalized_qposes_with_scripted_gctrl(env, episode):
    qposes = [ts.observation.qpos for ts in episode]
    gctrls = [ts.observation.gctrl for ts in episode]
    for qpos, gctrl in zip(qposes, gctrls):
        qpos[6 + 0] = env.task.BVXtask.puppet.position_normalize(gctrl[0])
        qpos[6 + 7] = env.task.BVXtask.puppet.position_normalize(gctrl[2])
    return qposes


def replay(env, qposes):
    ts = env.reset()
    episode_replay = [ts]
    for t in range(len(qposes)):
        action = qposes[t]
        ts = env.step(action)
        episode_replay.append(ts)
    return episode_replay


def create_data_dict(cfg, episode_replay, qposes):
    data_dict = {
        "/observations/qpos": [],
        "/observations/qvel": [],
        "/action": [],
    }
    for cam_name in cfg.camera_names:
        data_dict[f"/observations/images/{cam_name}"] = []

    while qposes:
        action = qposes.pop(0)
        ts = episode_replay.pop(0)
        obs = ts.observation
        data_dict["/observations/qpos"].append(obs.qpos)
        data_dict["/observations/qvel"].append(obs.qvel)
        data_dict["/action"].append(action)
        for cam_name in cfg.camera_names:
            data_dict[f"/observations/images/{cam_name}"].append(obs.images[cam_name])
    return data_dict


def create_hdf5_dataset(dataset_path, data_dict):
    with h5py.File(dataset_path + ".hdf5", "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = True
        obs = root.create_group("observations")
        image = obs.create_group("images")
        for cam_name in cfg.robot.camera_names:
            image.create_dataset(
                cam_name,
                (cfg.episode_len, 480, 640, 3),
                dtype="uint8",
                chunks=(1, 480, 640, 3),
            )
        obs.create_dataset("qpos", (cfg.episode_len, 14))
        obs.create_dataset("qvel", (cfg.episode_len, 14))
        root.create_dataset("action", (cfg.episode_len, 14))
        for name, array in data_dict.items():
            root[name][...] = array


def save_episode(i_eps, episode_replay, qposes):
    data_dict = create_data_dict(cfg.robot, episode_replay, qposes)
    dataset_dir = P.data_dir / cfg.task.name
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_path = os.path.join(dataset_dir, f"episode_{i_eps}")
    create_hdf5_dataset(dataset_path, data_dict)
