"""
Remember to change number of channels when db is changed
"""

from numpy.lib.function_base import select
from torch.utils.data import IterableDataset, RandomSampler
from torch.utils.data.dataloader import default_collate
import numpy as np
from framework.dataset.segmentation_db import Segmentation_db
from tqdm import tqdm
import torch
from torch.nn import functional as F
import sys
import cv2
from collections import deque


def label_to_outputs(label, channels=19):
    height, width = label.shape
    return cv2.resize(
        label, dsize=(width // 8 + 1, height // 8 + 1), interpolation=cv2.INTER_NEAREST
    )
    # if not isinstance(label, torch.Tensor):
    #     label = torch.from_numpy(label)
    # height, width = label.size()
    # labels_clone = F.interpolate(label.unsqueeze(1).float(), size=(height//8 + 1, width//8 + 1)).view(-1)
    # mask = (labels_clone != 255)
    # labels_filtered = labels_clone[mask]
    # return F.one_hot(labels_filtered.long(), channels)


class Buffer_db(IterableDataset):
    def __init__(self, initial_db, batch_size, domain="source", channels=19):
        """
        Buffer stored into memory

        """
        self.channels = channels
        self.distribution = np.zeros(channels)
        self.buffer = deque()
        print("Loading data to memory")
        for i in tqdm(range(len(initial_db))):
            sample = initial_db[i]
            sample["domain"] = domain
            sample["stored_predictions"] = sample["label"]
            self.buffer.append(sample)
        self.batch_size = batch_size
        self.type_dict = {}
        # type of each variable
        for key, val in self.buffer[0].items():
            self.type_dict[key] = type(val)
        self.pos = 0
        self.permutation = np.random.permutation(len(self.buffer))

    def __next__(self):
        items = []
        for _ in range(self.batch_size):
            items.append(self.buffer[self.pos])
            self.pos += 1
            self.pos %= len(self)
            if self.pos == 0:
                self.permutation = np.random.permutation(len(self.buffer))
        # items = np.random.choice(self.buffer, self.batch_size, replace=False)
        return default_collate(items)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return default_collate([self.buffer[self.permutation[index]]])

    def __iter__(self):
        return self

    def sequential(self):
        for i in range(len(self)):
            yield self[i]

    def add(self, item, policy="queue"):
        if policy == "queue":
            self.buffer.popleft()
            self.buffer.append(item)
        elif policy == "random":
            index = np.random.randint(len(self.buffer))
            self.buffer[index] = item
        else:
            raise NotImplementedError(f"the policy {policy}, has not been implemented")

    def __sizeof__(self) -> int:
        """
        The measurements suggest 0.5MB per sample,
        that leards to approximately 2.5GB for the whole Cityscapes dataset.
        """
        return get_size(self.buffer)

    def add_from_batch(self, batch, index, domain="target"):
        batch["domain"] = domain
        submited_batch = {}
        for key in self.type_dict.keys():
            sample = batch[key][index]
            if type(sample) != self.type_dict[key]:
                sample = sample.cpu().numpy()
            submited_batch[key] = sample
        self.add(submited_batch)


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size
