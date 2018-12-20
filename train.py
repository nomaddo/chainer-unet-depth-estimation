import os
import argparse
import random
import json

import chainer
from chainer import training
from chainer.training import extensions
from chainer import functions as F
from chainer import links as L
from chainer.datasets import split_dataset_random

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
        reshape = F.reshape(coarse, (batch_size, 1, 128, 128))
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
            self.c0 = L.Convolution2D(3, 32, 3, 1, 1)
            self.c1 = L.Convolution2D(32, 64, 4, 2, 1)
            self.c2 = L.Convolution2D(64, 64, 3, 1, 1)
            self.c3 = L.Convolution2D(64, 128, 4, 2, 1)
            self.c4 = L.Convolution2D(128, 128, 3, 1, 1)
            self.c5 = L.Convolution2D(128, 256, 4, 2, 1)
            self.c6 = L.Convolution2D(256, 256, 3, 1, 1)
            self.c7 = L.Convolution2D(256, 512, 4, 2, 1)
            self.c8 = L.Convolution2D(512, 512, 3, 1, 1)

            self.dc8  = L.Deconvolution2D(1024, 512, 4, 2, 1)
            self.dc7  = L.Convolution2D(512, 256, 3, 1, 1)
            self.dc6  = L.Deconvolution2D(512, 256, 4, 2, 1)
            self.dc5  = L.Convolution2D(256, 128, 3, 1, 1)
            self.dc4  = L.Deconvolution2D(256, 128, 4, 2, 1)
            self.dc3  = L.Convolution2D(128, 64, 3, 1, 1)
            self.dc2  = L.Deconvolution2D(128, 64, 4, 2, 1)
            self.dc1  = L.Convolution2D(64, 32, 3, 1, 1)
            self.dc0  = L.Convolution2D(64, 3, 3, 1, 1)
            self.final = L.Convolution2D(3, 1, 1)

            self.bnc0 = L.BatchNormalization(32)
            self.bnc1 = L.BatchNormalization(64)
            self.bnc2 = L.BatchNormalization(64)
            self.bnc3 = L.BatchNormalization(128)
            self.bnc4 = L.BatchNormalization(128)
            self.bnc5 = L.BatchNormalization(256)
            self.bnc6 = L.BatchNormalization(256)
            self.bnc7 = L.BatchNormalization(512)
            self.bnc8 = L.BatchNormalization(512)

            self.bnd8 = L.BatchNormalization(512)
            self.bnd7 = L.BatchNormalization(256)
            self.bnd6 = L.BatchNormalization(256)
            self.bnd5 = L.BatchNormalization(128)
            self.bnd4 = L.BatchNormalization(128)
            self.bnd3 = L.BatchNormalization(64)
            self.bnd2 = L.BatchNormalization(64)
            self.bnd1 = L.BatchNormalization(32)


    def forward(self, x):
        e0 = F.relu(self.bnc0(self.c0(x)))
        e1 = F.relu(self.bnc1(self.c1(e0)))
        e2 = F.relu(self.bnc2(self.c2(e1)))
        e3 = F.relu(self.bnc3(self.c3(e2)))
        e4 = F.relu(self.bnc4(self.c4(e3)))
        e5 = F.relu(self.bnc5(self.c5(e4)))
        e6 = F.relu(self.bnc6(self.c6(e5)))
        e7 = F.relu(self.bnc7(self.c7(e6)))
        e8 = F.relu(self.bnc8(self.c8(e7)))

        d8 = F.relu(self.bnd8(self.dc8(F.concat([e7, e8]))))
        d7 = F.relu(self.bnd7(self.dc7(d8)))
        d6 = F.relu(self.bnd6(self.dc6(F.concat([e6, d7]))))
        d5 = F.relu(self.bnd5(self.dc5(d6)))
        d4 = F.relu(self.bnd4(self.dc4(F.concat([e4, d5]))))
        d3 = F.relu(self.bnd3(self.dc3(d4)))
        d2 = F.relu(self.bnd2(self.dc2(F.concat([e2, d3]))))
        d1 = F.relu(self.bnd1(self.dc1(d2)))
        d0 = self.dc0(F.concat([e0, d1]))
        return F.sigmoid(self.final(d0))

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


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12345, help='seed for numpy cupy random module %(default)s')
    parser.add_argument("--device", type=int, default=7, help='%(default)s')
    parser.add_argument("--model_name", type=str, default="resnet50", help='model arch for training simple cnn, resnet50 or vgg16 default = %(default)s')
    parser.add_argument("--multiplier", type=float, default=1.0, help='%(default)s')
    parser.add_argument("--batch-size", type=int, default=16, help='%(default)s')
    parser.add_argument("--destination", type=str, default="trained", help='%(default)s')
    parser.add_argument("--resume", type=str, default="", help="default is empty string '' ")
    parser.add_argument("--epoch", type=int, default=100, help='num epoch for training %(default)s')
    args = parser.parse_args()
    return args




def train(args=None):
    if args.model_name == 'resnet50':
        model = RefinedNetwork(ResNet50)
        resize = True
    elif args.model_name == 'unet':
        model = UNET()
        resize = False
    else:
        assert False

    dataset = DepthDataset('./', resize=resize, augmentation=True, pca_lighting=True, crop=True)
    train, test = split_dataset_random(dataset, int(0.9 * len(dataset)), seed=args.seed)

    train_iter = chainer.iterators.MultiprocessIterator(train, args.batch_size, n_processes=4)
    test_iter = chainer.iterators.MultiprocessIterator(
        test, args.batch_size, repeat=False, shuffle=False)

    opt = chainer.optimizers.Adam()  # MomentumSGD(0.1)
    opt.setup(model)

    updater = training.StandardUpdater(train_iter, opt, device=args.device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='./result')

    snapshot_interval = (1, 'epoch')
    trainer.extend(extensions.Evaluator(test_iter, model,
                                        device=args.device), trigger=snapshot_interval)
    trainer.extend(extensions.ProgressBar(), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=snapshot_interval))
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_epoch_{.updater.epoch}.npz'), trigger=(10, 'epoch'))
    trainer.extend(extensions.PrintReport(
        ["epoch", "main/loss", 'validation/main/loss']), trigger=snapshot_interval)

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

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
