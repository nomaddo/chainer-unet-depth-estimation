from typing import Tuple, Dict, List

import copy
import os
import csv
import logging
from PIL import Image

from chainer.dataset import DatasetMixin
from chainercv import transforms
import chainercv
from scipy.ndimage.interpolation import rotate

import numpy as np
from numpy import float32

from pdb import set_trace

logger = logging.getLogger(__name__)


def read_jpg(path, dtype=float32):
    return chainercv.utils.read_image(path, dtype=dtype)


def read_png(path, dtype=float32):
    pic = Image.open(path)
    mode = pic.mode
    pic = np.asarray(pic, dtype=dtype)[np.newaxis]

    if mode == 'L':
        pic /= 255
    elif mode == 'I':
        pic /= 2 ** 32
    else:
        assert False
    return pic


def get_images(tuple: Tuple[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    pic, estimated = tuple

    jpg = read_jpg(pic)
    png = read_png(estimated)

    return jpg, png


class DepthDataset(DatasetMixin):

    def __init__(self, csv_dir: str, train=True, resize=True, augmentation=False, use_cache=False, dev=False) -> None:
        # self.base: List[Tuple[str, str]] = []
        self.base = []
        self.train = train
        self.model_name = 'depth'
        self.resize = resize
        self.use_cache = use_cache
        if use_cache:
            # self.cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
            self.cache = {}
        self.augmentation = augmentation

        cnt = 0
        with open(csv_dir + 'train.csv') as f:
            reader = csv.reader(f)
            for row in reader:
                pic = row[0]
                estimated = row[1]
                for dir in [pic, estimated]:
                    if not(os.path.exists(dir)):
                        raise Exception(dir + ' not found')
                if estimated == 'data/raw/ae9532af8fd548fea3e6b5b62d57a518_d0_2.png':
                    logger.info('BLACKLIST: ' + estimated)
                else:
                    self.base.append((pic, estimated))
                    cnt += 1
                    if dev and cnt > 100:
                        break

    def __len__(self) -> int:
        return len(self.base)

    def get_example(self, i: int):
        tuple = self.base[i]

        if self.use_cache:
            if i in self.cache:
                images = self.cache[i]
            else:
                images = get_images(tuple)
                self.cache[i] = images
        else:
            images = get_images(tuple)
        return images


if __name__ == '__main__':
    dataset = DepthDataset('./')
    print(dataset[0])
