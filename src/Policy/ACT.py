import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from omegaconf import OmegaConf

from detr.main import build_ACT_model_and_optimizer


class ACTPolicy(nn.Module):

    def __init__(self, robot_cfg, policy_cfg, chunk_size):
        super().__init__()
        overrides = OmegaConf.create(
            {
                "camera_names": list(robot_cfg.camera_names),
                "num_queries": int(chunk_size),
                "dA": int(robot_cfg.info.dA),
                "kl_weight": int(policy_cfg.kl_weight),
            }
        )
        if "optimizer" in policy_cfg:
            # optional: allow ACT.yaml to override lr / weight_decay / lr_backbone
            for k in ("lr", "lr_backbone", "weight_decay"):
                if k in policy_cfg.optimizer:
                    overrides[k] = policy_cfg.optimizer[k]

        self.model, self.optimizer = build_ACT_model_and_optimizer(overrides)
        self.kl_weight = float(policy_cfg.kl_weight)
        self.image_normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        print(f"KL Weight {self.kl_weight}")

    def forward(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        image = self.image_normalizer(image)
        if actions is not None:  # training / validation
            a_hat, _, (mu, logvar), _, _ = self.model(
                qpos, image, env_state, actions, is_pad
            )
            total_kld, _, _ = kl_divergence(mu, logvar)
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()

            loss_dict = {
                "l1": l1,
                "kl": total_kld[0],
                "loss": l1 + total_kld[0] * self.kl_weight,
            }
            return loss_dict

        # inference: sample from prior
        a_hat, _, (_, _), _, _ = self.model(qpos, image, env_state)
        return a_hat

    def configure_optimizers(self):
        return self.optimizer

    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise_kld, mean_kld
