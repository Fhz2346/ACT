from tqdm import tqdm
import torch.nn.functional as F
import torch
import pickle

from load import load_data
from load.util import norm_stats_to_dict
from omegaconf import OmegaConf
from Pathes import P
from utils import set_seed, repeater
from Network.cnn_mlp import CNNMLP, build_optimizer
from validate import Validator

if __name__ == "__main__":
    cfg = OmegaConf.load(P.config_dir / "imitate_episodes" / "CNNMLP.yaml")
    cfg.task = OmegaConf.load(P.config_dir / f"Task/{cfg.task.name}.yaml")
    cfg.robot = OmegaConf.load(P.config_dir / f"{cfg.robot.name}.yaml")

    P.ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = P.data_dir / cfg.task.name
    tra_dataloader, val_dataloader, stats = load_data([f"{dataset_dir}_scripted", f"{dataset_dir}_human"], cfg)
    with open(P.ckpt_dir / "dataset_stats.pkl", "wb") as f:
        pickle.dump(norm_stats_to_dict(*stats), f)

    tra_dataloader = repeater(tra_dataloader)
    validator = Validator(val_dataloader)
    set_seed(cfg.seed)

    policy = CNNMLP(cfg.robot, cfg.policy).cuda()
    policy.train()
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_params / 1e6,))

    optimizer = build_optimizer(policy, cfg.optimizer)
    pbar = tqdm(range(cfg.steps.total + 1))
    for step in pbar:
        optimizer.zero_grad()
        data = next(tra_dataloader)
        image, qpos, action = [d.cuda() for d in data]
        action_hat = policy(qpos, image)
        loss = F.mse_loss(action, action_hat)
        loss.backward()
        optimizer.step()

        if step % cfg.steps.every.validate == 0:
            val_loss = validator.validate(policy, step)
            pbar.set_postfix(val_loss=val_loss)
        if (step + 1) % cfg.steps.every.save == 0:
            torch.save(policy.state_dict(), P.ckpt_dir / f"{cfg.task.name}_{cfg.policy.name}_{step+1}.pth")
