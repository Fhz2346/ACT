import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F


class Validator:

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.min_val_loss = np.inf
        self.best_ckpt_info = None

    def validate(self, policy: torch.nn.Module, step: int):
        with torch.inference_mode():
            policy.eval()
            losses = []
            for batch_idx, data in enumerate(self.dataloader):
                image, qpos, action = [d.cuda() for d in data]
                action_hat = policy(qpos, image)
                loss = F.mse_loss(action, action_hat)
                losses.append(loss.item())
                if batch_idx > 50:
                    break
            val_loss = np.mean(losses)
            if val_loss < self.min_val_loss:
                self.min_val_loss = val_loss
                self.best_ckpt_info = (step, self.min_val_loss, deepcopy(policy.state_dict()))
        return val_loss
