"""Train ACT policy (chunked action prediction), mirroring train.py for CNNMLP."""

from tqdm import tqdm
import torch
import pickle
import numpy as np
from copy import deepcopy

from load import load_data
from load.util import norm_stats_to_dict
from omegaconf import OmegaConf
from Pathes import P
from utils import set_seed, repeater
from Policy import ACTPolicy


def validate(policy, dataloader, max_batches=50):
    with torch.inference_mode():
        policy.eval()
        losses = []
        for batch_idx, data in enumerate(dataloader):
            image, qpos, actions, is_pad = [d.cuda() for d in data]
            loss_dict = policy(qpos, image, actions, is_pad)
            losses.append(loss_dict["loss"].item())
            if batch_idx >= max_batches:
                break
        val_loss = float(np.mean(losses))
    policy.train()
    return val_loss


if __name__ == "__main__":
    cfg = OmegaConf.load(P.config_dir / "imitate_episodes" / "ACT.yaml")
    cfg.task = OmegaConf.load(P.config_dir / f"Task/{cfg.task.name}.yaml")
    cfg.robot = OmegaConf.load(P.config_dir / f"{cfg.robot.name}.yaml")

    P.ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = P.data_dir / cfg.task.name
    tra_dataloader, val_dataloader, stats = load_data(
        [f"{dataset_dir}_scripted", f"{dataset_dir}_human"], cfg
    )
    with open(P.ckpt_dir / "dataset_stats.pkl", "wb") as f:
        pickle.dump(norm_stats_to_dict(*stats), f)

    tra_dataloader = repeater(tra_dataloader)
    set_seed(cfg.seed)

    policy = ACTPolicy(cfg.robot, cfg.policy, cfg.chunk_size).cuda()
    policy.train()
    optimizer = policy.configure_optimizers()

    min_val_loss = np.inf
    best_state_dict = None
    best_step = None

    pbar = tqdm(range(cfg.steps.total + 1))
    for step in pbar:
        optimizer.zero_grad()
        image, qpos, actions, is_pad = [d.cuda() for d in next(tra_dataloader)]
        loss_dict = policy(qpos, image, actions, is_pad)
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()

        if step % cfg.steps.every.validate == 0:
            val_loss = validate(policy, val_dataloader)
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                l1=f"{loss_dict['l1'].item():.4f}",
                kl=f"{loss_dict['kl'].item():.4f}",
                val_loss=f"{val_loss:.4f}",
            )
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_step = step
                best_state_dict = deepcopy(policy.state_dict())

        if (step + 1) % cfg.steps.every.save == 0:
            ckpt_path = P.ckpt_dir / f"{cfg.task.name}_{cfg.policy.name}_{step + 1}.pth"
            torch.save(policy.state_dict(), ckpt_path)

    torch.save(policy.state_dict(), P.ckpt_dir / f"{cfg.task.name}_{cfg.policy.name}_last.pth")
    if best_state_dict is not None:
        torch.save(
            best_state_dict,
            P.ckpt_dir / f"{cfg.task.name}_{cfg.policy.name}_best_step_{best_step}.pth",
        )
        print(f"Best val loss {min_val_loss:.6f} at step {best_step}")
