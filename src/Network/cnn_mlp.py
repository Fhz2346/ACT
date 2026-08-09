import torch
from torch import nn
from .module.backbone import build_backbone
from torchvision import transforms

class MLP(nn.Module):

    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNNMLP(nn.Module):

    def __init__(self, robot_cfg, cfg):
        super().__init__()
        self.n_cam = len(robot_cfg.camera_names)
        self.image_normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.action_dim = robot_cfg.info.dA

        backbones = []
        for _ in range(self.n_cam):
            backbone = build_backbone(cfg.backbone)
            backbones.append(backbone)
        self.backbones = nn.ModuleList(backbones)

        backbone_down_projs = []
        for backbone in backbones:
            down_proj = nn.Sequential(
                nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                nn.Conv2d(128, 64, kernel_size=5),
                nn.Conv2d(64, 32, kernel_size=5)
            )
            backbone_down_projs.append(down_proj)
        self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

        dImlp = 768 * len(backbones) + robot_cfg.info.dS
        self.mlp = MLP([dImlp] + cfg.mlp.dHs + [robot_cfg.info.dA])

    def forward_image(self, image):
        bs = image.shape[0]
        image = self.image_normalizer(image)  # [bs, n_cam, n_chn, H, W]
        cam_features = []
        for cam_id in range(self.n_cam):
            features, _ = self.backbones[cam_id](image[:, cam_id])
            feature = features[0]
            feature = self.backbone_down_projs[cam_id](feature)
            feature = feature.reshape([bs, -1])
            cam_features.append(feature)
        overall_cam_feature = torch.cat(cam_features, axis=1)
        return overall_cam_feature

    def forward(self, qpos, image):
        overall_cam_feature = self.forward_image(image)
        features = torch.cat([overall_cam_feature, qpos], axis=1)
        action = self.mlp(features)
        return action


def build_optimizer(model, cfg):
    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": cfg.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return optimizer
