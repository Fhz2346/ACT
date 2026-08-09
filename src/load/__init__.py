import numpy as np
import torch
from torch.utils.data import DataLoader

from .util import flatten_list, get_norm_stats
from .EpisodicDataset import EpisodicDataset
from .EpisodicChunkDataset import EpisodicChunkDataset
from .fs import find_all_hdf5, load_all_hdf5


def get_all_norm_stats(dataset_path_list):
    all_qpos, all_action, all_eps_len = load_all_hdf5(dataset_path_list)
    action_ns = get_norm_stats(torch.cat(all_action, dim=0))
    qpos_ns = get_norm_stats(torch.cat(all_qpos, dim=0))
    return (action_ns, qpos_ns), all_eps_len


def BatchSampler(batch_size, eps_len_l):
    sum_dataset_len_l = np.cumsum([0] + [np.sum(eps_len) for eps_len in eps_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            eps_idx = np.random.choice(len(eps_len_l))
            step_idx = np.random.randint(sum_dataset_len_l[eps_idx], sum_dataset_len_l[eps_idx + 1])
            batch.append(step_idx)
        yield batch


def load_data(dataset_dir_l, cfg):
    dataset_path_ll = [find_all_hdf5(dataset_dir) for dataset_dir in dataset_dir_l]
    dataset_path_l = flatten_list(dataset_path_ll)
    norm_stats, all_eps_len = get_all_norm_stats(dataset_path_l)

    def get_train_val_split(train_ratio):
        num_eps_0 = len(dataset_path_ll[0])
        num_eps_l = [len(dataset_path_l) for dataset_path_l in dataset_path_ll]
        num_eps_cumsum = np.cumsum(num_eps_l)
        # obtain train test split on dataset_dir_l[0]
        shuffled_eps_ids_0 = np.random.permutation(num_eps_0)
        split_idx = int(train_ratio * num_eps_0)
        tra_eps_ids_0 = shuffled_eps_ids_0[:split_idx]
        val_eps_ids_0 = shuffled_eps_ids_0[split_idx:]
        tra_eps_ids_l = [tra_eps_ids_0] + [num_eps_cumsum[idx] + np.arange(num_eps) for idx, num_eps in enumerate(num_eps_l[1:])]
        val_eps_ids_l = [val_eps_ids_0]
        tra_eps_lens_l = [[all_eps_len[i] for i in tra_eps_ids] for tra_eps_ids in tra_eps_ids_l]
        val_eps_lens_l = [[all_eps_len[i] for i in val_eps_ids] for val_eps_ids in val_eps_ids_l]
        return tra_eps_ids_l, val_eps_ids_l, tra_eps_lens_l, val_eps_lens_l

    tra_eps_ids_l, val_eps_ids_l, tra_eps_lens_l, val_eps_lens_l = get_train_val_split(cfg.train_ratio)
    tra_eps_ids = np.concatenate(tra_eps_ids_l)
    val_eps_ids = np.concatenate(val_eps_ids_l)
    tra_eps_lens = flatten_list(tra_eps_lens_l)
    val_eps_lens = flatten_list(val_eps_lens_l)

    batch_sampler_tra = BatchSampler(cfg.batch_size, tra_eps_lens_l)
    batch_sampler_val = BatchSampler(cfg.batch_size, val_eps_lens_l)
    
    if cfg.chunk_size == 1:
        tra_dataset = EpisodicDataset(dataset_path_l, tra_eps_ids, tra_eps_lens, norm_stats, cfg.robot.camera_names)
        val_dataset = EpisodicDataset(dataset_path_l, val_eps_ids, val_eps_lens, norm_stats, cfg.robot.camera_names)
    else:
        tra_dataset = EpisodicChunkDataset(dataset_path_l, tra_eps_ids, tra_eps_lens, norm_stats, cfg.robot.camera_names, cfg.chunk_size)
        val_dataset = EpisodicChunkDataset(dataset_path_l, val_eps_ids, val_eps_lens, norm_stats, cfg.robot.camera_names, cfg.chunk_size)
    
    tra_dataloader = DataLoader(tra_dataset, batch_sampler=batch_sampler_tra, pin_memory=True, num_workers=2, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=True, num_workers=2, prefetch_factor=2)
    return tra_dataloader, val_dataloader, norm_stats
