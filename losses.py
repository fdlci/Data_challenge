import torch
from torch.autograd import Variable


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = (y_pred_f * y_true_f).sum()
    smooth = 0.0001
    return 1 - (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


def change_shape(y_true, y_pred, numLabels):
    encoded_target = y_pred.data.clone().zero_()
    encoded_target[...] = 0
    encoded_target.scatter_(1, torch.tensor(y_true.unsqueeze(1), dtype=torch.int64), 1.)
    encoded_target = Variable(encoded_target)
    return encoded_target


def dice_coef_multilabel(y_pred, y_true, numLabels):
    dice = 0
    y_true = change_shape(y_true, y_pred, numLabels)
    for index in range(numLabels):
        dice += dice_coef(y_true[:, index, :, :], y_pred[:, index, :, :])
    return dice / numLabels  # taking average
