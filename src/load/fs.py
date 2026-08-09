import os
import fnmatch
import h5py
import numpy as np
import torch

def find_all_hdf5(dataset_dir):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files


def parse_hdf5(hdf5_file):
    try:
        with h5py.File(hdf5_file, 'r') as root:
            qpos = root['/observations/qpos'][()]
            # qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
            T = action.shape[0]
            dummy_base_action = np.zeros([T, 2])
            action = np.concatenate([action, dummy_base_action], axis=-1)
    except Exception as e:
        print(f'Error loading {hdf5_file} in get_norm_stats')
        print(e)
        quit()
    return qpos, action


def load_all_hdf5(dataset_path_list):
    all_qpos_data = []
    all_action_data = []
    all_eps_len = []
    for dataset_path in dataset_path_list:
        qpos, action = parse_hdf5(dataset_path)
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_eps_len.append(len(qpos))
    return all_qpos_data, all_action_data, all_eps_len
