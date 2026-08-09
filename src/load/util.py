import numpy as np
import torch
from dataclasses import dataclass


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


@dataclass
class NormStats:
    mean: torch.Tensor
    std: torch.Tensor
    min: torch.Tensor
    max: torch.Tensor


def get_norm_stats(
    data: torch.Tensor,
    std_clip_range=(1e-2, np.inf),
    eps=0.0001,
):
    mean = data.mean(dim=[0]).float()
    std = data.std(dim=[0]).float()
    std = torch.clip(std, std_clip_range[0], std_clip_range[1])
    min = data.min(dim=0).values.float() - eps
    max = data.max(dim=0).values.float() + eps
    return NormStats(mean, std, min, max)


def normalize(data, ns: NormStats):
    return (data - ns.mean) / ns.std


def unnormalize(data, ns: NormStats):
    return data * ns.std + ns.mean


def norm_stats_to_dict(action_ns: NormStats, qpos_ns: NormStats):
    return {
        "action_mean": action_ns.mean.detach().cpu().numpy(),
        "action_std": action_ns.std.detach().cpu().numpy(),
        "qpos_mean": qpos_ns.mean.detach().cpu().numpy(),
        "qpos_std": qpos_ns.std.detach().cpu().numpy(),
    }


def dict_to_norm_stats(stats: dict):
    def _ns(mean_key, std_key):
        mean = torch.from_numpy(np.asarray(stats[mean_key])).float()
        std = torch.from_numpy(np.asarray(stats[std_key])).float()
        return NormStats(mean=mean, std=std, min=mean.clone(), max=mean.clone())

    return _ns("action_mean", "action_std"), _ns("qpos_mean", "qpos_std")
