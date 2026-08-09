import numpy as np
import torch
import h5py

from .util import normalize


class EpisodicDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path_l, eps_ids, eps_lens, norm_stats, camera_names):
        super(EpisodicDataset).__init__()
        self.dataset_path_l = dataset_path_l

        self.eps_ids = eps_ids
        self.eps_lens = eps_lens
        self.cum_len = np.cumsum(eps_lens)
        self.max_eps_len = max(eps_lens)

        self.action_ns, self.qpos_ns = norm_stats
        self.camera_names = camera_names

    def __len__(self):
        return self.cum_len[-1]

    def _locate_transition(self, index):
        assert index < self.cum_len[-1]
        eps_index = np.argmax(self.cum_len > index) # argmax returns first True index
        start_ts = index - (self.cum_len[eps_index] - self.eps_lens[eps_index])
        eps_id = self.eps_ids[eps_index]
        return eps_id, start_ts

    def _parse_hdf5(self, dataset_path, ts):
        try:
            with h5py.File(dataset_path, 'r') as root:
                action = root['/action'][()][ts]
                dummy_base_action = np.zeros([2])
                action = np.concatenate([action, dummy_base_action], axis=-1)
                qpos = root["/observations/qpos"][ts]
                qvel = root["/observations/qvel"][ts]
                images = []
                for cam_name in self.camera_names:
                    images.append(root[f'/observations/images/{cam_name}'][ts])
                images = np.stack(images, axis=0)
                images = images.transpose(0, 3, 1, 2)
                return qpos, qvel, images, action
        except:
            print(f'Error loading {dataset_path} in _parse_hdf5')
            quit()

    def __getitem__(self, index):
        eps_id, ts = self._locate_transition(index)
        dataset_path = self.dataset_path_l[eps_id]
        qpos, qvel, images, action = self._parse_hdf5(dataset_path, ts)

        action = torch.from_numpy(action).float()
        action = normalize(action, self.action_ns)

        images = torch.from_numpy(images)
        images = images / 255.0

        qpos = torch.from_numpy(qpos).float()
        qpos = normalize(qpos, self.qpos_ns)
        return images, qpos, action
