import numpy as np
import LCD
import torch

def mIOU(label, pred, num_classes=10):
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    label = label.view(-1)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # and ignore_index is num_classes, thus ignore_index is
    # not considered in computation of IoU.
    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else:
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    return np.mean(present_iou_list)


def class_weight():
    weights = np.zeros((LCD.N_CLASSES,))
    num_ign_classes = len(LCD.IGNORED_CLASSES_IDX)
    weights[num_ign_classes:] = (1 / LCD.TRAIN_CLASS_COUNTS[2:]) * LCD.TRAIN_CLASS_COUNTS[2:].sum() / (
                LCD.N_CLASSES - 2)
    weights[LCD.IGNORED_CLASSES_IDX] = 0.

    class_weights = torch.FloatTensor(weights)
    return class_weights


def epsilon_kl_divergence(y_true, y_pred):
    class_distribution_true = np.apply_along_axis(np.bincount, axis=1, arr=y_true.flatten(1), minlength=LCD.N_CLASSES)
    class_distribution_pred = np.apply_along_axis(np.bincount, axis=1, arr=y_pred.flatten(1), minlength=LCD.N_CLASSES)
    # Normalize to sum to 1
    normalized_class_distribution_true = (class_distribution_true.T/class_distribution_true.sum(1)).T
    normalized_class_distribution_pred = (class_distribution_pred.T/class_distribution_pred.sum(1)).T
    # add a small constant for smoothness around 0
    normalized_class_distribution_true += 1e-7
    normalized_class_distribution_pred += 1e-7

    score = np.mean(np.sum(normalized_class_distribution_true * np.log(normalized_class_distribution_true / normalized_class_distribution_pred), 1))
    try:
        assert np.isfinite(score)
    except AssertionError as e:
        raise ValueError('score is NaN or infinite') from e
    return score