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
# Local
from LCD import LandCoverData as LCD
from metrics import *
from losses import *


def training(model, train_loader, valid_loader, data_sizes, epochs, optimizer, criterion, scheduler, title, device):
    print("Device", device)
    model.to(device)

    training_loss = []
    validation_loss = []

    best_model = copy.deepcopy(model.state_dict())
    best_kl = 1000

    loaders = {"train": train_loader, "valid": valid_loader}
    since = time.time()
    for epoch in range(1, epochs + 1):
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
                image, mask = image.to(device), mask.to(device)
                with torch.set_grad_enabled(phase == 'train'):

                    output = model(image)
                    _, preds = torch.max(output, 1)

                    loss = criterion(output, mask)

                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                running_iou += mIOU(mask, preds)
                running_kl_div += epsilon_kl_divergence(mask.cpu(), preds.cpu())

            epoch_loss = running_loss / data_sizes[phase]
            epoch_iou = running_iou / data_sizes[phase]
            epoch_kl = running_kl_div / data_sizes[phase]
            if phase == 'train':
                scheduler.step()
                training_loss.append(epoch_loss)
            else:
                validation_loss.append(epoch_loss)

            print('{} Loss: {:.4f} IoU: {:.4f} KL_div: {:.4f}'.format(phase, epoch_loss, epoch_iou, epoch_kl))

            if phase == 'valid' and epoch_kl < best_kl:
                best_kl = epoch_kl
                best_model = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "unet_timm-efficientnet-b3_3_channels.pt")

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
    plt.show()  # Change title for every model

    return model


def train_model(loader_train, loader_valid, data_sizes, model, epochs, lr, device):
    optimizer_ft = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.BCELoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    # Training the model
    title = 'Variations of the training and validation loss SGD'
    model_ft = training(model, loader_train, loader_valid, data_sizes, epochs, optimizer_ft, criterion,
                        exp_lr_scheduler, title, device)

    return model_ft
