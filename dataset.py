from typing import Tuple, Dict, List

import copy
import os
import csv
import logging
from PIL import Image

from chainer.dataset import DatasetMixin
from chainercv import transforms
import chainercv

import numpy as np
from numpy import float32

logger = logging.getLogger(__name__)


def get_images(tuple: Tuple[str, str], resize=False) -> Tuple[np.ndarray, np.ndarray]:
    pic, estimated = tuple
    jpg = chainercv.utils.read_image(pic, dtype=float32)
    png = chainercv.utils.read_image(estimated, dtype=float32, color=False)

    if resize:
        jpg = chainercv.transforms.resize(jpg, (128, 128))
        png = chainercv.transforms.resize(png, (128, 128))

    png /= 255.
    return jpg, png


class DepthDataset(DatasetMixin):

    def __init__(self, csv_dir: str, train=True, resize=True, augmentation=False, pca_lighting=False, crop=False) -> None:
        # self.base: List[Tuple[str, str]] = []
        self.base = []
        self.train = train
        self.model_name = 'depth'
        self.resize = resize
        # self.cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.cache = {}
        self.augmentation = augmentation
        self.pca_lighting = pca_lighting
        self.crop = crop

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
            images = get_images(tuple, resize=self.resize)
            self.cache[i] = images

        input, answer = copy.deepcopy(images)

        if self.augmentation:
            input = transforms.random_flip(input, x_random=True)
            answer = transforms.random_flip(answer, x_random=True)

        if self.pca_lighting:
            input = transforms.pca_lighting(input, sigma=5)

        if self.crop:
            H = input.shape[1]
            W = input.shape[2]
            scale_w, scale_h = np.random.uniform(1.0, 3.0, 2)
            resize_w, resize_h = int(W * scale_w), int(H * scale_h)

            input = transforms.resize(input, (resize_h, resize_w))
            input = transforms.random_crop(input, (H, W))
            answer = transforms.resize(answer, (resize_h, resize_w))
            answer = transforms.random_crop(answer, (H, W))

        images = (input, answer)

        if self.train:
            return images
        else:
            return images[0]


if __name__ == '__main__':
    dataset = DepthDataset('./')
    print(dataset[0])
