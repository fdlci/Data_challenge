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

    def reset(self, param=None):
        """
        Resets the class by sending the model to the device and initializes the logger.
        params:
        param: If not None, it will load the state dictionary from the param argument.
        """
        self.model.to(self.device)
        if param:
            self.model.load_state_dict(param)
        for key in ['loss', 'kl_div', 'iou']:
            self.logger[key] = {}
            for mode in ['train', 'val']:
                self.logger[key][mode] = []

    def execute(self, mode='train'):
        assert mode in ['train', 'val'], "mode should belong to ['train', 'val']"

        running_loss = 0
        running_iou = 0
        running_kl_div = 0

        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        for image, mask in tqdm(self.loaders[mode]):
            image, mask = image.to(self.device), mask.to(self.device)
            output = self.model(image)
            _, preds = torch.max(output, 1)

            loss, bce_loss, dice_loss = self.criterion(output, mask.squeeze())

            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            with torch.no_grad():
                running_loss += loss.item()
                running_iou += mIOU(mask, preds)
                running_kl_div += epsilon_kl_divergence(mask.cpu(), preds.cpu())

        self.logger['loss'][mode].append(running_loss / len(self.loaders[mode]))
        self.logger['iou'][mode].append(running_iou / len(self.loaders[mode]))
        self.logger['kl_div'][mode].append(running_kl_div / len(self.loaders[mode]))
        if mode == 'val' and running_kl_div / len(self.loaders[mode]) < self.best_kl:
            self.best_kl = running_kl_div / len(self.loaders[mode])
            self.best_model_params = copy.deepcopy(self.model.state_dict())

    def run(self, epochs, verbose=True):
        """
        Runs training for specified nnumber of epochs.
        params:
        epochs (int): Number of epochs
        verbose (bool): If true, it prints the metrics for each epoch, otherwise, nothing is printed.
        """
        for e in range(epochs):
            self.execute('train')
            self.execute('val')
            if self.scheduler:
                self.scheduler.step()
            if verbose:
                print(f"Epoch {e+1}/{epochs} Training Loss:", self.logger['loss']['train'][e], "Training IoU:",
                      self.logger['iou']['train'][e], "Training KL:", self.logger['kl_div']['train'][e])
                print(f"Epoch {e+1}/{epochs} Validation Loss:", self.logger['loss']['val'][e], "Validation IoU:",
                      self.logger['iou']['val'][e], "Validation KL:", self.logger['kl_div']['val'][e])

    def get_best_model(self):
        """
        Returns the parameters of the best model according to the kl divergence.
        """
        return self.best_model_params

    def plot(self, epoch_min=0, epoch_max=-1, figsize=(20, 10)):
        """
        Plots the evolution of different metrics.
        params:
        epoch_min (int): Where to start the plot
        epoch_max (int): Where to stop the plot
        figsize (tuple): Tuple for figure size.
        """
        fig, ax = plt.subplots(1, 3, figsize=figsize)
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
