import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
# from segmentation_models_pytorch.losses import DiceLoss
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from _functional import soft_dice_score, to_tensor
from constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE


def one_hot(y_true, y_pred, numLabels):
    encoded_target = y_pred.data.clone().zero_()
    encoded_target[...] = 0
    encoded_target.scatter_(1, torch.tensor(y_true.unsqueeze(1), dtype=torch.int64), 1.)
    encoded_target = Variable(encoded_target)
    return encoded_target


class DiceLoss(_Loss):

    def __init__(
            self,
            mode: str,
            classes: Optional[List[int]] = None,
            log_loss: bool = False,
            from_logits: bool = True,
            smooth: float = 0.0,
            ignore_index: Optional[int] = None,
            eps: float = 1e-7,
    ):
        """Implementation of Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases
        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error 
                (denominator will be always greater or equal to eps)
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)
        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(DiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
            y_true = y_true.permute(0, 2, 1)  # H, C, H*W

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

        scores = soft_dice_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return loss.mean()


# taken from https://github.com/asanakoy/kaggle_carvana_segmentation/blob/master/asanakoy/losses.py
class BCELoss2d(nn.Module):
    """
    Binary Cross Entropy loss function
    """

    def __init__(self):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        num_labels = logits.shape[1]
        logits_flat = logits.view(-1)
        if labels.size(1) != num_labels:
            labels = one_hot(labels, logits, num_labels)
        labels_flat = labels.view(-1)
        return self.bce_loss(logits_flat, labels_flat)


class WeightedBCELoss2d(nn.Module):
    def __init__(self):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, logits, labels, weights):
        num_labels = logits.shape[1]
        w = weights.view(-1)
        if labels.size(1) != num_labels:
            labels = one_hot(labels, logits, num_labels)
        logits = logits.view(-1)
        gt = labels.view(-1)
        # http://geek.csdn.net/news/detail/126833
        loss = logits.clamp(min=0) - logits * gt + torch.log(1 + torch.exp(-logits.abs()))
        loss = loss * w
        loss = loss.sum() / w.sum()
        return loss


class CrossEntropyLoss(nn.Module):
    """
    Binary Cross Entropy loss function
    """

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        bs, num_classes = logits.shape[0], logits.shape[1]
        logits_flat = logits.view(bs, num_classes, -1)
        labels_flat = labels.view(bs, -1)
        return self.ce_loss(logits_flat, labels_flat)


class CombinedLoss(nn.Module):
    def __init__(self, cross_entropy=True, alpha=1, is_log_dice=False):
        super(CombinedLoss, self).__init__()
        self.is_log_dice = is_log_dice
        self.alpha = alpha
        self.cross_entropy = cross_entropy
        self.bce = BCELoss2d()
        self.ce = CrossEntropyLoss()
        self.soft_dice = DiceLoss("multiclass", smooth=1e-4)

    def forward(self, logits, labels):
        if self.cross_entropy:
            loss = self.ce(logits, labels)
        else:
            loss = self.bce(logits, labels)
        dice_loss = self.soft_dice(logits, labels)

        if self.is_log_dice:
            total_loss = self.alpha * loss - (1 - dice_loss).log()
        else:
            total_loss = self.alpha * loss + dice_loss
        return total_loss, loss, dice_loss

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
