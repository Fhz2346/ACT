"""Run a trained ACT policy in joint-space sim_env."""

import pickle
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from Pathes import P
from env import make_sim_env
from load import load_data
from load.util import normalize, unnormalize, norm_stats_to_dict, dict_to_norm_stats
from Policy import ACTPolicy


def load_cfg():
    cfg = OmegaConf.load(P.config_dir / "imitate_episodes" / "ACT.yaml")
    cfg.task = OmegaConf.load(P.config_dir / f"Task/{cfg.task.name}.yaml")
    cfg.robot = OmegaConf.load(P.config_dir / f"{cfg.robot.name}.yaml")
    if "sim" not in cfg:
        demo_cfg = OmegaConf.load(P.config_dir / "demonstration.yaml")
        cfg.sim = demo_cfg.sim
    if "episode_len" not in cfg:
        demo_cfg = OmegaConf.load(P.config_dir / "demonstration.yaml")
        cfg.episode_len = demo_cfg.episode_len
    return cfg


def get_image(ts, camera_names):
    images = []
    for cam_name in camera_names:
        image = ts.observation.images[cam_name]  # HWC uint8
        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0  # CHW
        images.append(image)
    image = np.stack(images, axis=0)
    return torch.from_numpy(image).float().cuda().unsqueeze(0)  # [1, n_cam, C, H, W]


def resolve_ckpt_path(cfg):
    ckpt_name = cfg.get("eval_ckpt_name", None)
    if ckpt_name is not None:
        ckpt_path = P.ckpt_dir / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    pattern = f"{cfg.task.name}_{cfg.policy.name}_*.pth"
    ckpts = sorted(P.ckpt_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoint matching {pattern} in {P.ckpt_dir}. "
            "Train first with train1.py or set eval_ckpt_name in ACT.yaml."
        )
    return ckpts[-1]


def load_policy(cfg, ckpt_path):
    policy = ACTPolicy(cfg.robot, cfg.policy, cfg.chunk_size).cuda()
    state_dict = torch.load(ckpt_path, map_location="cuda")
    if hasattr(policy, "deserialize"):
        status = policy.deserialize(state_dict)
        print(status)
    else:
        policy.load_state_dict(state_dict)
    policy.eval()
    print(f"Loaded policy: {ckpt_path}")
    return policy


def load_or_build_stats(cfg):
    stats_path = P.ckpt_dir / "dataset_stats.pkl"
    if stats_path.exists():
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        print(f"Loaded stats: {stats_path}")
        return dict_to_norm_stats(stats)

    print("dataset_stats.pkl not found, computing from dataset...")
    dataset_dir = P.data_dir / cfg.task.name
    _, _, stats = load_data([f"{dataset_dir}_scripted", f"{dataset_dir}_human"], cfg)
    P.ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "wb") as f:
        pickle.dump(norm_stats_to_dict(*stats), f)
    print(f"Saved stats: {stats_path}")
    return stats


def save_gif(frames, path, fps):
    if not frames:
        return
    from pathlib import Path
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in frames]
    duration_ms = max(1, int(round(1000.0 / fps)))
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"Saved GIF: {path} ({len(frames)} frames, {fps:.1f} fps)")


def rollout(env, policy, cfg, qpos_ns, action_ns, onscreen=False, save_gif_path=None):
    """Open-loop execute an action chunk every `chunk_size` steps (no temporal_agg)."""
    env.task.randomize_initial_env()
    ts = env.reset()
    rewards = []
    frames = []

    gif_camera = cfg.get("gif_camera", "top")
    gif_stride = int(cfg.get("gif_stride", 1))
    DT = float(cfg.sim.DT)
    query_frequency = int(cfg.chunk_size)
    all_actions = None
    record_gif = save_gif_path is not None

    if onscreen:
        import matplotlib.pyplot as plt

        ax = plt.subplot()
        plt_img = ax.imshow(env.physics.render(height=480, width=640, camera_id=gif_camera))
        plt.ion()

    for t in range(int(cfg.episode_len)):
        t0 = time.time()
        frame = env.physics.render(height=480, width=640, camera_id=gif_camera)
        if onscreen:
            plt_img.set_data(frame)
            plt.pause(DT)
        if record_gif and (t % gif_stride == 0):
            frames.append(frame.copy())

        qpos = np.asarray(ts.observation.qpos, dtype=np.float32)
        qpos_t = normalize(torch.from_numpy(qpos), qpos_ns).cuda().unsqueeze(0)

        if t % query_frequency == 0:
            image_t = get_image(ts, cfg.robot.camera_names)
            if t == 0:
                with torch.inference_mode():
                    for _ in range(10):
                        policy(qpos_t, image_t)
                print("network warm up done")
            with torch.inference_mode():
                all_actions = policy(qpos_t, image_t)  # [1, chunk_size, dA]

        raw_action = all_actions[:, t % query_frequency].squeeze(0).cpu()
        action = unnormalize(raw_action, action_ns).numpy()
        target_qpos = action[:-2]  # drop dummy base_action -> 14-dim

        ts = env.step(target_qpos)
        rewards.append(ts.reward)

        sleep_t = DT - (time.time() - t0)
        if sleep_t > 0:
            time.sleep(sleep_t)

    if onscreen:
        import matplotlib.pyplot as plt

        plt.close()

    if record_gif:
        fps = float(cfg.sim.get("FPS", 1.0 / DT)) / max(gif_stride, 1)
        save_gif(frames, save_gif_path, fps=fps)

    rewards = np.array([r for r in rewards if r is not None], dtype=np.float64)
    episode_return = float(rewards.sum()) if len(rewards) else 0.0
    highest_reward = float(rewards.max()) if len(rewards) else 0.0
    success = highest_reward == env.task.max_reward
    return success, episode_return, highest_reward


def main():
    cfg = load_cfg()
    onscreen = bool(cfg.get("onscreen_render", False))
    num_rollouts = int(cfg.get("num_rollouts", 10))
    save_gif_flag = bool(cfg.get("save_gif", False))

    ckpt_path = resolve_ckpt_path(cfg)
    action_ns, qpos_ns = load_or_build_stats(cfg)
    policy = load_policy(cfg, ckpt_path)
    env = make_sim_env(cfg)

    from pathlib import Path

    gif_dir = Path(cfg.get("gif_dir") or (P.ckpt_dir / "gifs"))
    if save_gif_flag:
        gif_dir.mkdir(parents=True, exist_ok=True)

    successes, returns, highest_rewards = [], [], []
    for i in tqdm(range(num_rollouts), desc="eval ACT"):
        gif_path = None
        if save_gif_flag:
            gif_path = gif_dir / f"{cfg.task.name}_{cfg.policy.name}_rollout{i:03d}.gif"
        success, episode_return, highest_reward = rollout(
            env,
            policy,
            cfg,
            qpos_ns,
            action_ns,
            onscreen=onscreen,
            save_gif_path=gif_path,
        )
        successes.append(success)
        returns.append(episode_return)
        highest_rewards.append(highest_reward)
        tqdm.write(
            f"Rollout {i}: return={episode_return:.1f}, "
            f"max_reward={highest_reward:.0f}/{env.task.max_reward}, success={success}"
        )

    success_rate = float(np.mean(successes))
    avg_return = float(np.mean(returns))
    print(f"\nSuccess rate: {success_rate:.2%} ({sum(successes)}/{num_rollouts})")
    print(f"Average return: {avg_return:.2f}")
    for r in range(env.task.max_reward + 1):
        n = int((np.array(highest_rewards) >= r).sum())
        print(f"Reward >= {r}: {n}/{num_rollouts} = {n / num_rollouts:.1%}")
    if save_gif_flag:
        print(f"GIFs saved under: {gif_dir}")


if __name__ == "__main__":
    main()
