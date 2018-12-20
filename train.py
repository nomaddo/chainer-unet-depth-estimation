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
    parser.add_argument("--batch-size", type=int, default=64, help='%(default)s')
    parser.add_argument("--destination", type=str, default="trained", help='%(default)s')
    parser.add_argument("--resume", type=str, default="", help="default is empty string '' ")
    parser.add_argument("--epoch", type=int, default=100, help='num epoch for training %(default)s')
    args = parser.parse_args()
    return args


def train(args=None):
    dataset = DepthDataset('./', augmentation=True)
    train, test = split_dataset_random(dataset, int(0.9 * len(dataset)), seed=args.seed)

    if args.model_name == 'resnet50':
        model = RefinedNetwork(ResNet50)
    else:
        assert False

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
