import random

import numpy as np
import torch

PMIN = 1e-5

def set_random_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DSCLoss:

    def __init__(self, smoothing=1):
        super(DSCLoss, self).__init__()

        self.smoothing = smoothing

    def __call__(self, yhat, y):

        p = yhat.sigmoid()
        p = p.view(-1, 1)
        y = y.view(1, -1)
        numer = 2 * y @ p
        denom = y.sum() + p.sum()
        numer = numer + self.smoothing
        denom = denom + self.smoothing
        dsc_loss = numer / denom

        return 1 - dsc_loss

class BoundedNLLLoss:

    def __init__(self, pmin=PMIN):
        self.loss_fn = torch.nn.NLLLoss()
        self.pmin = pmin
    
    def __call__(self, yhat, y):
        loss = self.loss_fn(yhat, y)
        return (1. / (np.log(1. / self.pmin))) * loss

def compute_01(scores, y, average=True):
    """
    Assumes first dim is the batch dim.
    Assumes second dim is the score dim.
    """
    incorrect = (scores.argmax(1) != y).long().sum().float().item()
    if average:
        return incorrect / y.size(0)
    else:
        return incorrect

def compute_dsc(scores, y, average=True):
    """
    implementation adapted from the following:
    # gist.github.com/the-bass/cae9f3976866776dea17a5049013258d

    Assumes first dim is the batch dim.
    Assumes second dim is the score dim.
    Computes dsc per image in batch and averages.
    """
    yhat = (scores.sigmoid() > .5).long()
    dsc = 0.
    for j in range(y.size(0)):
        confusion_vector = yhat[j].float() / y[j].float()
        # Element-wise division of the 2 tensors returns a new
        # tensor which holds a unique value for each case:
        #   1     :prediction and truth are 1 (True Positive)
        #   inf   :prediction is 1 and truth is 0 (False Positive)
        #   nan   :prediction and truth are 0 (True Negative)
        #   0     :prediction is 0 and truth is 1 (False Negative)
        TP = torch.sum(confusion_vector == 1).item()
        FP = torch.sum(torch.isinf(confusion_vector)).item()
        TN = torch.sum(torch.isnan(confusion_vector)).item()
        FN = torch.sum(confusion_vector == 0).item()
        dsc += 2 * TP / (2 * TP + FP + FN) if TP != 0 else 0.
    if average:
        return dsc / y.size(0)
    else:
        return dsc