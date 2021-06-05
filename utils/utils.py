import sys
import yaml
import numpy as np


def accuracy(predictions, gt):
    m = gt.shape[0]
    acc = np.sum(predictions == gt) / m
    return acc


def is_debug_session():
    gettrace = getattr(sys, 'gettrace', None)
    debug_session = not ((gettrace is None) or (not gettrace()))
    return debug_session


def load_config_yml(config_file):
    try:
        with open(config_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            config = config['config']
            return config
    except:
        print('Config file {} is missing'.format(config_file))
        exit(1)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


# PyTorch Optimization tutorial
# https://www.youtube.com/watch?v=9mS1fIYj1So
def zero_grad(model):
    for p in model.parameters():
        p.grad = None


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
