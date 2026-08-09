from tqdm import tqdm

from env import make_ee_sim_env, make_sim_env
from utils import *

pbar = tqdm(range(cfg.num_episodes))
for i_eps in pbar:
    pbar.set_description(f"Rollout: {i_eps:03d}")
    ee_env = make_ee_sim_env(cfg)
    ee_env.task.randomize_initial_env()
    episode = rollout_ee(ee_env)
    success, episode_return = evaluate_episode(ee_env, episode)
    qposes = get_normalized_qposes_with_scripted_gctrl(ee_env, episode)
    # Replaying joint commands
    env = make_sim_env(cfg)
    episode_replay = replay(env, qposes)
    success, episode_return = evaluate_episode(env, episode_replay)
    save_episode(i_eps, episode_replay, qposes)
    pbar.set_postfix({"success": success, "return": episode_return})
