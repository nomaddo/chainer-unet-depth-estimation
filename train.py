import os
import argparse
import random
import json
import copy

import scipy
import chainer
from chainer import training
from chainer.training import extensions
from chainer import functions as F
from chainer import links as L
from chainer.datasets import split_dataset_random
import chainercv
from chainercv import transforms

import numpy as np
try:
    import cupy as xp
except ImportError:
    import numpy as xp

from dataset import DepthDataset


class ResNet50(chainer.Chain):
    def __init__(self, **kwargs):
        super(ResNet50, self).__init__()
        with self.init_scope():
            self.base = L.ResNet50Layers()
            self.fc_1 = L.Linear(None, 1024)
            self.fc_2 = L.Linear(None, 128 * 128)

    def __call__(self, x):
        h = self.base(x, layers=["pool5"])["pool5"]
        h = F.relu(self.fc_1(h))
        h = self.fc_2(h)
        return h

    def disable_target_layers(self):
        disables = ['conv1',
                    'res2',
                    'res3',
                    'res4',
                    # 'res5',
                    ]

        for layer in disables:
            self.base[layer].disable_update()


class Conv(chainer.Chain):
    def __init__(self, n_in, n_out, ksize, stride, pad):
        super(Conv, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(n_in, n_out, ksize, stride, pad)
            self.bn = L.BatchNormalization(n_out)

    def __call__(self, x):
        return F.relu(self.bn(self.conv(x)))


class Deconv(chainer.Chain):
    def __init__(self, n_in, n_out, ksize, stride, pad):
        super(Deconv, self).__init__()
        with self.init_scope():
            self.conv = L.Deconvolution2D(n_in, n_out, ksize, stride, pad)
            self.bn = L.BatchNormalization(n_out)

    def __call__(self, x):
        return F.relu(self.bn(self.conv(x)))


class RefinedNetwork(chainer.Chain):

    def __init__(self, f):
        super(RefinedNetwork, self).__init__()
        with self.init_scope():
            self.coarse = f()
            self.fine1 = L.Convolution2D(None, 63, 9, stride=1, pad=4)
            self.fine3 = L.Convolution2D(None, 1, 5, stride=1, pad=2)
            self.fine4 = L.Convolution2D(None, 1, 5, stride=1, pad=2)

    def forward(self, x):
        batch_size = x.shape[0]
        fine1 = F.max_pooling_2d(self.fine1(x), 3, stride=1, pad=1)
        coarse = self.coarse(x)
        reshape = F.reshape(coarse, (batch_size, 1, 240, 320))
        fine2 = F.concat((fine1, reshape), axis=1)
        fine3 = self.fine3(fine2)
        return F.sigmoid(self.fine4(fine3))

    def __call__(self, x, t):
        output = self.forward(x)
        h = F.mean_squared_error(t, output)

        self.loss = h
        chainer.reporter.report({'loss': self.loss}, self)
        return h


class UNET(chainer.Chain):

    def __init__(self):
        super(UNET, self).__init__()
        with self.init_scope():
            rate = 4
            self.c0 = Conv(3, 8 * rate, 3, 1, 1)
            self.c1 = Conv(None, 16 * rate, 4, 2, 1)
            self.c2 = Conv(None, 16 * rate, 3, 1, 1)
            self.c3 = Conv(None, 32 * rate, 4, 2, 1)
            self.c4 = Conv(None, 32 * rate, 3, 1, 1)
            self.c5 = Conv(None, 64 * rate, 4, 2, 1)
            self.c6 = Conv(None, 64 * rate, 3, 1, 1)
            self.c7 = Conv(None, 128 * rate, 4, 2, 1)
            self.c8 = Conv(None, 128 * rate, 3, 1, 1)

            self.dc8 = Deconv(1024, 512, 4, 2, 1)
            self.dc7 = Conv(None, 256, 3, 1, 1)
            self.dc6 = Deconv(None, 256, 4, 2, 1)
            self.dc5 = Conv(None, 128, 3, 1, 1)
            self.dc4 = Deconv(None, 128, 4, 2, 1)
            self.dc3 = Conv(None, 64, 3, 1, 1)
            self.dc2 = Deconv(None, 64, 4, 2, 1)
            self.dc1 = Conv(None, 32, 3, 1, 1)
            self.dc0 = Conv(None, 3, 3, 1, 1)
            self.final = L.Convolution2D(None, 1, 1)

    def forward(self, x):
        e0 = self.c0(x)
        e1 = self.c1(e0)
        e2 = self.c2(e1)
        e3 = self.c3(e2)
        e4 = self.c4(e3)
        e5 = self.c5(e4)
        e6 = self.c6(e5)
        e7 = self.c7(e6)
        e8 = self.c8(e7)

        d8 = self.dc8(F.concat([e7, e8]))
        d7 = self.dc7(d8)
        d6 = self.dc6(F.concat([e6, d7]))
        d5 = self.dc5(d6)
        d4 = self.dc4(F.concat([e4, d5]))
        d3 = self.dc3(d4)
        d2 = self.dc2(F.concat([e2, d3]))
        d1 = self.dc1(d2)
        d0 = self.dc0(F.concat([e0, d1]))
        final = F.sigmoid(self.final(d0))
        return final

    def __call__(self, x, t):
        h = self.forward(x)
        loss = F.mean_squared_error(h, t)
        self.loss = loss
        chainer.reporter.report({'loss': self.loss}, self)
        return loss


def save_args(args):
    if not os.path.exists(args.destination):
        os.mkdir(args.destination)

    for k, v in vars(args).items():
        print(k, v)

    with open(os.path.join(args.destination, "args.json"), 'w') as f:
        json.dump(vars(args), f)


def parse_bool(s: str) -> bool:
    if s == 'True' or 'true' or '1':
        return True
    elif s == 'False' or 'false' or '0':
        return False
    else:
        raise Exception('cannot parse: {}'.format(s))


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12345, help='seed for numpy cupy random module %(default)s')
    parser.add_argument("--device", type=int, default=7, help='%(default)s')
    parser.add_argument("--model_name", type=str, default="resnet50", help='model arch for training simple cnn, resnet50 or vgg16 default = %(default)s')
    parser.add_argument("--batch-size", type=int, default=16, help='%(default)s')
    parser.add_argument("--destination", type=str, default="trained", help='%(default)s')
    parser.add_argument("--resume", type=str, default="", help="default is empty string '' ")
    parser.add_argument("--epoch", type=int, default=100, help='num epoch for training %(default)s')
    parser.add_argument("--augmentation", type=parse_bool, default=True, help='augmentation: %(default)s')
    parser.add_argument("--dev", type=parse_bool, default=False, help='development option: %(default)s')
    args = parser.parse_args()
    return args


def resize(images):
    input, answer = images
    input = chainercv.transforms.resize(input, (240, 320))
    answer = chainercv.transforms.resize(answer, (240, 320))

    images = (input, answer)
    return images


DEG_RANGE = np.linspace(-20, 20, 100)


def rotate_image(image, deg, expand=False):
    return scipy.ndimage.rotate(image, deg, axes=(2, 1), reshape=expand)


def transform(images):
    input, answer = copy.deepcopy(images)

    # rotate
    deg = np.random.choice(DEG_RANGE)
    input = rotate_image(input, deg)
    answer = rotate_image(answer, deg)

    # resize
    H, W = input.shape[1:]
    h_resize = int(np.random.uniform(240, H * 2.0))
    w_resize = int(np.random.uniform(320, W * 2.0))
    input = chainercv.transforms.resize(input, (h_resize, w_resize))
    answer = chainercv.transforms.resize(answer, (h_resize, w_resize))

    # crop
    input, slice = transforms.random_crop(input, (240, 320), return_param=True)
    answer = answer[:, slice["y_slice"], slice['x_slice']]

    # flip
    input, param = transforms.random_flip(input, x_random=True, return_param=True)
    if param['x_flip']:
        transforms.flip(answer, x_flip=True)

    # pca_lighting:
    input = transforms.pca_lighting(input, sigma=5)

    return resize((input, answer))


def train(args=None):
    if args.model_name == 'resnet50':
        model = RefinedNetwork(ResNet50)
    elif args.model_name == 'unet':
        model = UNET()
    else:
        assert False

    if args.resume:
        chainer.serializers.load_npz(args.resume, model)

    dataset = DepthDataset('./', augmentation=args.augmentation, dev=args.dev)
    size = len(dataset)
    train, test = split_dataset_random(dataset, int(size * 0.95), seed=args.seed)
    train = chainer.datasets.TransformDataset(train, transform)
    test = chainer.datasets.TransformDataset(test, resize)

    train_iter = chainer.iterators.MultiprocessIterator(train, args.batch_size, n_processes=4)
    test_iter = chainer.iterators.MultiprocessIterator(
        test, args.batch_size, repeat=False, shuffle=False)

    opt = chainer.optimizers.Adam()  # MomentumSGD(0.1)
    opt.setup(model)

    updater = training.StandardUpdater(train_iter, opt, device=args.device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.destination)

    snapshot_interval = (1, 'epoch')
    trainer.extend(extensions.Evaluator(test_iter, model,
                                        device=args.device), trigger=snapshot_interval)
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.LogReport(trigger=snapshot_interval))
    trainer.extend(extensions.snapshot_object(
        model, 'model_epoch_{.updater.epoch}.npz'), trigger=(1, 'epoch'))
    trainer.extend(extensions.PrintReport(
        ["epoch", "main/loss", 'validation/main/loss']), trigger=snapshot_interval)
    trainer.run()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    xp.random.seed(seed)


def main():
    args = parse_argument()
    set_random_seed(args.seed)
    save_args(args)
    train(args)


if __name__ == '__main__':
    main()
