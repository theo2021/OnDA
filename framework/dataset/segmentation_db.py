from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
from os import path
from framework.utils.func import color_mapper
from framework.dataset.base_dataset import _load_img

from PIL import Image

base_transform = lambda mean, std: transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean / 255, std / 255)]
)


class Segmentation_db(Dataset):
    def __init__(
        self,
        root_folder,
        pandas_metadata,
        class_map,
        image_size,
        labels_size=None,
        transforms=base_transform(0, 255),
        predictions_path="tmp_predictions",
        original_label=False,
    ):
        """
        Mean, Variance should be in the range of [0,255]

        """
        self.metadata = pandas_metadata
        self.root = root_folder
        self.image_size = image_size
        if type(class_map) is dict:
            self.map = color_mapper(class_map)
        else:
            self.map = class_map
        if labels_size is None:
            labels_size = image_size
        self.labels_size = labels_size
        self.transforms = transforms
        try:
            if not path.exists(predictions_path):
                os.makedirs(predictions_path)
        except:
            print(
                "dataloader folder for saving prior predictions could not be created!"
            )
        self.prediction_path = predictions_path
        self.original_label = original_label

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        image_path = path.join(self.root, row["image_path"])
        soft_predictions_path = path.join(
            self.prediction_path, row["image_path"].replace(".png", "_proda.npy")
        )
        image = self.__get_image(image_path)
        output = {
            "image": self.preprocess(image),
            "image_path": image_path,
            "soft_path": soft_predictions_path,
        }
        if "label_path" in row.keys() and row["label_path"] is not None:
            label_path = path.join(self.root, row["label_path"])
            label = self.__get_label(label_path)
            label_raw = self.__get_label(label_path, True)
            label_resized = self.__get_label(label_path, resized=True)
            output["label"] = self.map(label).astype(np.uint8)
            output["label_path"] = label_path
            output["label_res"] = self.map(label_resized).astype(np.uint8)
            if self.original_label:
                output["label_raw"] = self.map(label_raw).astype(np.uint8)
            if path.exists(soft_predictions_path):
                output["soft_predictions"] = np.load(soft_predictions_path)
        return output

    def __get_image(self, image_path):
        return _load_img(image_path, self.image_size, Image.BICUBIC, rgb=True)

    def __get_label(self, label_path, original=False, resized=False):
        if original:
            return _load_img(label_path, None, Image.NEAREST, rgb=self.map.rgb)
        if resized:
            return _load_img(
                label_path,
                [int(x / 8 + 1) for x in self.labels_size],
                Image.NEAREST,
                rgb=self.map.rgb,
            )
        return _load_img(label_path, self.labels_size, Image.NEAREST, rgb=self.map.rgb)

    def preprocess(self, image):
        image = image[:, :, ::-1]  # change to BGR for Advent
        return self.transforms(image.copy())
