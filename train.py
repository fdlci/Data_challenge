# Essentials
import copy
import pandas as pd
from tqdm.notebook import tqdm
# Local
from metrics import *
from utils import *


class Trainer:
    def __init__(self, model, loaders, optimizer, criterion, scheduler=None, device=None):
        self.logger = {}
        self.model = model
        self.loaders = loaders
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

        self.best_model_params = copy.deepcopy(model.state_dict())
        self.best_kl = 1000
        self.total_epochs = 0

    def reset(self, param=None):
        """
        Resets the class by sending the model to the device and initializes the logger.
        params:
        param: If not None, it will load the state dictionary from the param argument.
        """
        self.total_epochs = 0
        self.model.to(self.device)
        if param:
            self.model.load_state_dict(param)
        for key in ['loss', 'kl_div', 'iou', 'bce_loss', 'dice_loss']:
            self.logger[key] = {}
            for mode in ['train', 'val']:
                self.logger[key][mode] = []

    def execute(self, mode='train'):
        assert mode in ['train', 'val'], "mode should belong to ['train', 'val']"

        running_loss = 0
        running_iou = 0
        running_kl_div = 0
        running_bce_loss = 0
        running_dice_loss = 0

        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        for image, mask in tqdm(self.loaders[mode]):
            image, mask = image.to(self.device), mask.to(self.device).to(torch.int64)
            output = self.model(image)
            _, preds = torch.max(output, 1)

            if mode == 'train':
                loss, bce_loss, dice_loss = self.criterion(output, mask.squeeze())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    loss, bce_loss, dice_loss = self.criterion(output, mask.squeeze())
            with torch.no_grad():
                running_loss += loss.item()
                running_iou += mIOU(mask, preds)
                running_kl_div += epsilon_kl_divergence(mask.cpu(), preds.cpu())
                running_bce_loss += bce_loss.item()
                running_dice_loss += dice_loss.item()

        self.logger['loss'][mode].append(running_loss / len(self.loaders[mode]))
        self.logger['iou'][mode].append(running_iou / len(self.loaders[mode]))
        self.logger['kl_div'][mode].append(running_kl_div / len(self.loaders[mode]))
        self.logger['bce_loss'][mode].append(running_bce_loss / len(self.loaders[mode]))
        self.logger['dice_loss'][mode].append(running_dice_loss / len(self.loaders[mode]))

        if mode == 'val' and running_kl_div / len(self.loaders[mode]) < self.best_kl:
            self.best_kl = running_kl_div / len(self.loaders[mode])
            self.best_model_params = copy.deepcopy(self.model.state_dict())

    def run(self, epochs, verbose=True, reduce_on_plateau=False):
        """
        Runs training for specified number of epochs.
        params:
        epochs (int): Number of epochs
        verbose (bool): If true, it prints the metrics for each epoch, otherwise, nothing is printed.
        """
        for e in range(epochs):
            self.execute('train')
            self.execute('val')
            if self.scheduler:
                if reduce_on_plateau:
                    self.scheduler.step(self.logger['kl_div']["val"][self.total_epochs])
                else:
                    self.scheduler.step()
            if verbose:
                print(f"Epoch {1 + self.total_epochs}/{epochs} Learning rate:", self.optimizer.param_groups[0]['lr'])

                print(f"Epoch {1 + self.total_epochs}/{epochs} Training Loss:",
                      self.logger['loss']['train'][self.total_epochs], "Training BCE Loss",
                      self.logger['bce_loss']['train'][self.total_epochs], "Training Dice Loss",
                      self.logger['dice_loss']['train'][self.total_epochs], "Training IoU:",
                      self.logger['iou']['train'][self.total_epochs], "Training KL:",
                      self.logger['kl_div']['train'][self.total_epochs])

                print(f"Epoch {1 + self.total_epochs}/{epochs} Validation Loss:",
                      self.logger['loss']['val'][self.total_epochs], "Validation BCE Loss",
                      self.logger['bce_loss']['val'][self.total_epochs], "Validation Dice Loss",
                      self.logger['dice_loss']['val'][self.total_epochs], "Validation IoU:",
                      self.logger['iou']['val'][self.total_epochs], "Validation KL:",
                      self.logger['kl_div']['val'][self.total_epochs])

            self.total_epochs += 1

    def get_best_model(self):
        """
        Returns the parameters of the best model according to the kl divergence.
        """
        print("Model has a best KL divergence of", self.best_kl)
        return self.best_model_params

    def plot(self, epoch_min=0, epoch_max=-1, figsize=(20, 10)):
        """
        Plots the evolution of different metrics.
        params:
        epoch_min (int): Where to start the plot
        epoch_max (int): Where to stop the plot
        figsize (tuple): Tuple for figure size.
        """
        fig, ax = plt.subplots(1, len(self.logger.keys()), figsize=figsize)
        for i, key in enumerate(self.logger):
            for mode in ["train", "val"]:
                ax[i].plot(self.logger[key][mode][epoch_min:epoch_max], label=mode)
            ax[i].set_xlabel("Epochs")
            ax[i].set_ylabel(key)
            ax[i].legend()
        plt.show()

    def generate_submission(self, loader, fname='submission'):
        """
        Generate a submission csv ready for the challenge.
        params:
        loader (DataLoader): DataLoader object for the dataset we're trying to generate predictions for
        fname (str): The filename of the output csv
        """
        sub_dict = {"sample_id": [], "no_data": [], "clouds": [], "artificial": [], "cultivated": [], "broadleaf": [],
                    "coniferous": [], "herbaceous": [], "natural": [], "snow": [], "water": []}
        for image, path in tqdm(loader):
            image = image.to(self.device)
            with torch.no_grad():
                output = self.model(image)
                _, preds = torch.max(output, 1)
                class_dis = batch_distribution(preds.cpu())
                sub_dict["sample_id"] += [int(p.split('.')[0]) for p in path]
                for key in LCD.CLASSES:
                    sub_dict[key] += class_dis[:, LCD.CLASSES.index(key)].tolist()
        df_sub = pd.DataFrame.from_dict(sub_dict)
        df_sub = df_sub.sort_values(by='sample_id')
        df_sub.to_csv(f'{fname}.csv')
        print(f"File is saved as {fname}.csv")
