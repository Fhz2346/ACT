import torch
import numpy as np
from itertools import repeat

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        # print(f'Epoch {epoch} done')
        epoch += 1
