from pathlib import Path
from copy import deepcopy
import numpy as np
from PIL import Image
from torch.utils import data


class BaseIterable(data.IterableDataset):
    def __init__(self, root, list_path, set_, image_size, labels_size, mean):
        self.root = Path(root)
        self.set = set_
        self.list_path = list_path.format(self.set)
        self.image_size = image_size
        if labels_size is None:
            self.labels_size = self.image_size
        else:
            self.labels_size = labels_size
        self.mean = mean
        with open(self.list_path) as f:
            self.img_ids = [i_id.strip() for i_id in f]
        self.files = []
        for name in self.img_ids:
            img_file, label_file = self.get_metadata(name)
            self.files.append((img_file, label_file, name))
        self.indexs = np.random.permutation(len(self.files))
        self.current_pos = 0
        self.epoch = 0
        self.stop = False  # stops when epoch ends

    def get_metadata(self, name):
        raise NotImplementedError

    def preprocess(self, image):
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        return image.transpose((2, 0, 1))

    def get_image(self, file):
        return _load_img(file, self.image_size, Image.BICUBIC, rgb=True)

    def get_labels(self, file):
        return _load_img(file, self.labels_size, Image.NEAREST, rgb=False)

    def get_sample(self, perc):
        """
        Removes a sample of the dataset and returns it
        Input: perc, the percentage of the data to be taken into the validation
        """
        choice = np.random.choice(self.indexs, np.round(int(len(self.indexs) * perc)))
        self.indexs = np.array(list(set(self.indexs) - set(choice)))
        replica = deepcopy(self)
        replica.stop = True
        replica.set = "val"
        replica.indexs = choice
        return replica

    def __next__(self):
        if self.current_pos >= len(self.indexs):
            if self.stop:
                raise StopIteration
            self.current_pos = 0
            self.indexs = np.random.permutation(self.indexs)
            self.epoch += 1
        index = self.indexs[self.current_pos]
        self.current_pos += 1
        return self[index]

    def __iter__(self):
        w_info = data.get_worker_info()
        if w_info is None:
            sub_indexs = self.indexs
        else:
            self.worker = w_info.id
            self.num_workers = w_info.num_workers
            sub_indexs = []
            for i in self.indexs:
                if i % self.num_workers == self.worker:
                    sub_indexs.append(i)
        replica = deepcopy(self)
        replica.indexs = np.array(sub_indexs)
        return replica


"""
Code provided by ADVENT, but changed
"""


def _load_img(file, size, interpolation, rgb):
    img = Image.open(file)
    if rgb:
        img = img.convert("RGB")
    if size is not None:
        img = img.resize(size, interpolation)
    return np.asarray(img, np.uint8)  # it was float32


class extended_list:
    def __init__(self, item_list, indexs):
        self.items = item_list
        self.indexs = indexs

    def __len__(self):
        return len(self.indexs)

    def __getitem__(self, i):
        return self.items[self.indexs[i]]


class BaseDataset(data.Dataset):
    def __init__(self, root, list_path, set_, max_iters, image_size, labels_size, mean):
        self.root = Path(root)
        self.set = set_
        self.list_path = list_path.format(self.set)
        self.image_size = image_size
        if labels_size is None:
            self.labels_size = self.image_size
        else:
            self.labels_size = labels_size
        self.mean = mean
        with open(self.list_path) as f:
            self.img_ids = [i_id.strip() for i_id in f]
        self.file_list = []
        for name in self.img_ids:
            img_file, label_file = self.get_metadata(name)
            self.file_list.append((img_file, label_file, name))
        index_list = []
        if max_iters is not None:
            for i in range(int(np.ceil(float(max_iters) / len(self.img_ids))) - 1):
                index_list.extend(np.random.permutation(len(self.file_list)).tolist())
            self.files = extended_list(self.file_list, index_list)
        else:
            self.files = self.file_list
        # if max_iters is not None:
        #     self.files
        #     self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

    def get_metadata(self, name):
        raise NotImplementedError

    def __len__(self):
        return len(self.files)

    def preprocess(self, image):
        image = image[:, :, ::-1]  # change to BGR
        image_tmp = image - self.mean
        return image_tmp.transpose((2, 0, 1))

    def get_image(self, file):
        return _load_img(file, self.image_size, Image.BICUBIC, rgb=True)

    def get_labels(self, file):
        return _load_img(file, self.labels_size, Image.NEAREST, rgb=False)
