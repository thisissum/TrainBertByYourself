import torch
from torch import nn


class SOPLoss(nn.Module):
    """A binary loss used for sentence order prediction
    """

    def __init__(self):
        super(SOPLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)


class MLMLoss(nn.Module):
    """A loss used for predict masked word, NLLoss used
    """

    def __init__(self, pad=0):
        super(MLMLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=pad)

    def forward(self, y_pred, y_true):
        """
        shape(y_pred) = batch_size, seq_len, class_num
        shape(y_true) = batch_size, class_num
        """
        return self.loss(y_pred.permute(0, 2, 1), y_true)
