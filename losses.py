import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn


def change_shape(y_true, y_pred, numLabels):
    encoded_target = y_pred.data.clone().zero_()
    encoded_target[...] = 0
    encoded_target.scatter_(1, torch.tensor(y_true.unsqueeze(1), dtype=torch.int64), 1.)
    encoded_target = Variable(encoded_target)
    return encoded_target


#
# def dice_coef(y_true, y_pred):
#     y_true_f = y_true.flatten()
#     y_pred_f = y_pred.flatten()
#     intersection = (y_pred_f * y_true_f).sum()
#     smooth = 0.0001
#     return 1 - (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)
#
#
# def dice_coef_multilabel(y_pred, y_true, numLabels):
#     dice = 0
#     y_true = change_shape(y_true, y_pred, numLabels)
#     for index in range(numLabels):
#         dice += dice_coef(y_true[:, index, :, :], y_pred[:, index, :, :])
#     return dice / numLabels  # taking average


class BCELoss2d(nn.Module):
    """
    Binary Cross Entropy loss function
    """

    def __init__(self):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        logits_flat = logits.view(-1)
        labels_flat = labels.view(-1)
        return self.bce_loss(logits_flat, labels_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self, num_labels):
        super(SoftDiceLoss, self).__init__()
        self.num_labels = num_labels

    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=1)
        if labels.size(1) != self.num_labels:
            labels = change_shape(labels, logits, self.num_labels)
        num = labels.size(0)
        score = 0
        for class_ in range(self.num_labels):
            m1 = probs[:, class_, ...].view(num, -1)
            m2 = labels[:, class_, ...].view(num, -1)
            intersection = (m1 * m2)
            score += (2. * (intersection.sum(1) + 1e-4) / (m1.sum(1) + m2.sum(1) + 1e-4)) / self.num_labels
        return 1 - score.mean()


class CombinedLoss(nn.Module):
    def __init__(self, is_log_dice=False):
        super(CombinedLoss, self).__init__()
        self.is_log_dice = is_log_dice

        self.bce = BCELoss2d()
        self.soft_dice = SoftDiceLoss()

    def forward(self, logits, labels):
        size = logits.size()
        assert size[1] == 1, size
        logits = logits.view(size[0], size[2], size[3])
        labels = labels.view(size[0], size[2], size[3])

        bce_loss = self.bce(logits, labels)
        dice_loss = self.soft_dice(logits, labels)

        if self.is_log_dice:
            l = bce_loss - (1 - dice_loss).log()
        else:
            l = bce_loss + dice_loss
        return l, bce_loss, dice_loss
