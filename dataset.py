from typing import Tuple, Dict, List

import copy
import os
import csv
import logging
from PIL import Image

from chainer.dataset import DatasetMixin
import chainercv

import numpy as np
from numpy import float32

logger = logging.getLogger(__name__)


def get_images(tuple: Tuple[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    pic, estimated = tuple
    jpg = chainercv.utils.read_image(pic, dtype=float32)
    png = chainercv.utils.read_image(estimated, dtype=float32, color=False)

    jpg = chainercv.transforms.resize(jpg, (128, 128))
    png = chainercv.transforms.resize(png, (128, 128))

    png /= 255.
    return jpg, png


class DepthDataset(DatasetMixin):

    def __init__(self, csv_dir: str, train=True, augmentation=False) -> None:
        # self.base: List[Tuple[str, str]] = []
        self.base = []
        self.train = train
        self.model_name = 'depth'
        # self.cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.cache = {}
        self.augmentation = augmentation

        with open(csv_dir + 'train.csv') as f:
            reader = csv.reader(f)
            for row in reader:
                pic = row[0]
                estimated = row[1]
                for dir in [pic, estimated]:
                    if not(os.path.exists(dir)):
                        raise Exception(dir + ' not found')

                self.base.append((pic, estimated))

    def __len__(self) -> int:
        return len(self.base)

    def get_example(self, i: int):
        tuple = self.base[i]

        if i in self.cache:
            images = self.cache[i]
        else:
            images = get_images(tuple)
            self.cache[i] = images

        input, ans = copy.deepcopy(images)

        if self.augmentation:
            aug_input  = chainercv.transforms.random_flip(input, x_random=True)
            aug_answer = chainercv.transforms.random_flip(ans, x_random=True)
            images = (aug_input, aug_answer)

        if self.train:
            return images
        else:
            return images[0]


if __name__ == '__main__':
    dataset = DepthDataset('./')
    print(dataset[0])
