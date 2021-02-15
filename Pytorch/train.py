# Essentials
import time
import copy
from collections import OrderedDict
import random
import os
from tifffile import TiffFile
from PIL import Image, ImageOps
from pathlib import Path
from torch._C import dtype
from tqdm.notebook import tqdm
# Sklearn
from sklearn.model_selection import train_test_split
# Data
import numpy as np
import pandas as pd
# Plot
import matplotlib.pyplot as plt
# Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
# Torchvision
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

from LCD import LandCoverData as LCD
from dataset import ImageSegementationDataset as dataset
from dataset import train_val_dataset
from Unet import UNet

def class_weight():
    weights = np.zeros((LCD.N_CLASSES,))
    num_ign_classes = len(LCD.IGNORED_CLASSES_IDX)
    weights[num_ign_classes:] = (1 / LCD.TRAIN_CLASS_COUNTS[2:])* LCD.TRAIN_CLASS_COUNTS[2:].sum() / (LCD.N_CLASSES-2)
    weights[LCD.IGNORED_CLASSES_IDX] = 0.

    class_weights = torch.FloatTensor(weights)
    return class_weights

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

def epsilon_kl_divergence(y_true, y_pred):
    
    y_true, y_pred = y_true.cpu(), y_pred.cpu()

    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)

    # Normalize to sum to 1 if it's not already
    y_true /= y_true.sum(1, keepdims=True)
    y_pred /= y_pred.sum(1, keepdims=True)
    # add a small constant for smoothness around 0
    y_true += 1e-7
    y_pred += 1e-7
    score = np.mean(np.sum(y_true * np.log(y_true / y_pred), 1))
    try:
        assert np.isfinite(score)
    except AssertionError as e:
        raise ValueError('score is NaN or infinite') from e
    return score

def training(model, train_loader, valid_loader, data_sizes, epochs, optimizer, scheduler, title):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device", device)
    model.to(device)
    
    since = time.time()

    training_loss = []
    validation_loss = []
    num_workers = 1

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    loaders = {"train": train_loader, "valid": valid_loader}
    print(data_sizes)
    step = 0

    # criterion = DiceLoss()
    class_weights = class_weight().to(device)
    criterion = nn.CrossEntropyLoss(class_weights)

    for epoch in range(1, epochs+1):
        print(f'Epoch {epoch}/{epochs}')
        print('-' * 10)

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_iou = 0
            running_kl_div = 0

            for image, mask in tqdm(loaders[phase]):
                image = image.to(device)
                mask = mask.type(torch.long).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):

                    output = model(image)
                    # print(output)
                    _, preds = torch.max(output, 1)
                    sourceTensor = mask.clone().detach()
                    loss = criterion(output, sourceTensor.squeeze())
                    # loss = criterion(output, mask)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                # running_corrects += torch.sum(iou_pytorch(output, mask))
                running_iou += mIOU(mask, preds)
                running_kl_div += epsilon_kl_divergence(mask, preds)

            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss/data_sizes[phase]
            epoch_iou = running_iou/data_sizes[phase]
            epoch_kl = running_kl_div/data_sizes[phase]
            if phase == 'train':
                training_loss.append(epoch_loss)
            else:
                validation_loss.append(epoch_loss)

            print('{} Loss: {:.4f} IoU: {:.4f} KL_div: {:.4f}'.format(phase, epoch_loss, epoch_iou, epoch_kl))
            
            if phase == 'valid' and epoch_iou > best_acc:
                best_acc = epoch_iou
                best_model = copy.deepcopy(model.state_dict())

            
    # Plotting the validation loss and training loss
    print('validation loss: ' + str(validation_loss))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model)

    # plot the training and validation loss
    plt.figure()
    plt.plot(training_loss, 'b', label='Training Loss')
    plt.plot(validation_loss, 'r', label='Validation Loss')
    plt.title(title)
    plt.legend()
    plt.show()

    return model

def train_model(train_dir, model, epochs, batch_size, lr):

    # loading the data
    train_set, val_set = train_val_dataset(dataset(train_dir), val_split=0.2)
    loader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    loader_valid = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    data_sizes = {"train": len(loader_train), "valid": len(loader_valid)}

    # Optimizing all parameters
    # optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(model.parameters(), lr = lr)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Training the model
    title = 'Variations of the training and validation loss'
    model_ft = training(model, loader_train, loader_valid, data_sizes, epochs, optimizer_ft, exp_lr_scheduler, title)

    return model_ft

if __name__ == '__main__':

    train_dir = 'Pytorch/Small_dataset/train'
    model = UNet()
    epochs = 10
    batch_size = 16
    lr = 1e-5

    trained_model = train_model(train_dir, model, epochs, batch_size, lr)
    torch.save(trained_model.state_dict(),"unet.pt")