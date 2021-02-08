import torch
import random
import numpy as np


class AverageMeter(object):
    """
    Computes and stores the average and current value of metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """
        String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


def partial_acc(pred, label, idx):
    """
    :param pred: predicted results of classifier, 1d integer tensor
    :param label: ground truth labels, 1d integer tensor
    :param idx: positions to be evaluated, 1d bool tensor
    :return: accuracy of the selected portions
    """
    total = idx.sum().item() + 1e-8
    correct = (pred[idx] == label[idx]).sum().item()
    acc = correct / total
    return acc


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
