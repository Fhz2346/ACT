import numpy as np
import torch
import h5py

from .EpisodicDataset import EpisodicDataset
from .util import normalize

class EpisodicChunkDataset(EpisodicDataset):

    def __init__(self, dataset_path_l, eps_ids, eps_lens, norm_stats, camera_names, chunk_size):
        super(EpisodicChunkDataset, self).__init__(dataset_path_l, eps_ids, eps_lens, norm_stats, camera_names)
        self.chunk_size = chunk_size

    def _parse_hdf5(self, dataset_path, start_ts):
        try:
            with h5py.File(dataset_path, 'r') as root:
                actions = root['/action'][()]
                eps_len = actions.shape[0]
                dummy_base_actions = np.zeros([eps_len, 2])
                actions = np.concatenate([actions, dummy_base_actions], axis=-1)
                actions = actions[start_ts:]
                # get observation at start_ts only
                qpos = root["/observations/qpos"][start_ts]
                qvel = root["/observations/qvel"][start_ts]
                images = []
                for cam_name in self.camera_names:
                    images.append(root[f'/observations/images/{cam_name}'][start_ts])
                images = np.stack(images, axis=0)
                images = images.transpose(0, 3, 1, 2)
                return qpos, qvel, images, actions
        except:
            print(f'Error loading {dataset_path} in _parse_hdf5')
            quit()

    def process_actions(self, actions):
        action_len, dA = actions.shape
        padded_actions = np.zeros((self.max_eps_len, dA), dtype=np.float32)
        padded_actions[:action_len] = actions
        actions = padded_actions[:self.chunk_size]
        actions = torch.from_numpy(actions).float()
        actions = normalize(actions, self.action_ns)

        is_pad = np.zeros(self.max_eps_len)
        is_pad[action_len:] = 1
        is_pad = is_pad[:self.chunk_size]
        is_pad = torch.from_numpy(is_pad).bool()
        return actions, is_pad

    def __getitem__(self, index):
        eps_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_l[eps_id]
        qpos, qvel, images, actions = self._parse_hdf5(dataset_path, start_ts)

        images = torch.from_numpy(images)
        images = images / 255.0

        actions, is_pad = self.process_actions(actions)
        
        qpos = torch.from_numpy(qpos).float()
        qpos = normalize(qpos, self.qpos_ns)
        return images, qpos, actions, is_pad
