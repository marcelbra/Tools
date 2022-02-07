from parser import Parser, WordEncoder, SpanEncoder
from utils import get_targets, load_data, save_errors
from Data import Data

from collections import defaultdict, Counter
from tqdm import tqdm
import pickle
import random
import sys

import torch.nn as nn
import torch.optim as optim
import torch

class Trainer:

    def __init__(self, paths, epochs, config):

        # Set paths
        self.path_model = paths["path_model"]
        self.path_parameters = paths["path_parameters"]
        self.path_errors = paths["path_errors"]

        # Set variable needed below
        self.config = config
        self.epochs = epochs
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Set validation and patience parameters
        self.validation_amount = config["validation_amount"] if config["validation_amount"] else 2
        self.patience = config["patience"] if config["patience"] else 3
        self.decay_from_step = config["decay_from_step"]

        # Store current best variables
        self.patience_counter = 0
        self.least_wrong = float("inf")
        self.best_model = None

    def train(self, model, data):

        optimizer = eval(self.config["optimizer"])
        optimizer = optimizer(params=model.parameters(), lr=self.config["lr"])

        scheduler = None
        if self.config["lr_decay"] is not None:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, self.config["lr_decay"])

        loss_func = nn.CrossEntropyLoss()
        model.to(self.device)

        for epoch in range(1, self.epochs + 1):
            print(f"\nStarting epoch: {epoch}")
            # Originally, `current_train_matrices` was saved to list holding all epochs
            # But now everything is dynamically printed to stdout
            current_train_metrics = self.do_epoch(epoch, model, data, loss_func, scheduler, optimizer)

        data.store_parameters(self.path_parameters)

    def do_epoch(self, epoch, model, data, loss_func, scheduler=None, optimizer=None):

        counter = 0
        epoch_metrics = Counter()
        curr_dataset = data.train_parses if optimizer else data.dev_parses
        random.shuffle(curr_dataset)
        n = len(curr_dataset)

        for i in tqdm(range(len((curr_dataset))), position=0, leave=True):

            epoch_metrics += self.do_step(model, curr_dataset[i], data, loss_func, optimizer)

            validate = (i+1) % int(n/self.validation_amount) == 0
            if optimizer and validate:
                counter, model = self.do_validation(counter, epoch, model, data, loss_func, epoch_metrics)

        if scheduler and epoch >= self.decay_from_step:
            if epoch == self.decay_from_step:
                print(f"\nStarting weight decay now with gamma={self.config['lr_decay']}.\n")
            scheduler.step()

        return epoch_metrics

    def do_validation(self, counter, epoch, model, data, loss_func, epoch_metrics):
        """
        Do validation `self.validation_amount` times per epoch and
        check if the model trained on the current portion of the training
        set has improved. If not, reset model to previos state.
        """

        counter += 1
        curr_valid_metrics = self.do_epoch(epoch, model, data, loss_func)
        curr_valid_wrong = curr_valid_metrics["wrong"]
        s = f"\nValidation run giving {curr_valid_wrong} wrong tags. "

        if curr_valid_wrong < self.least_wrong:
            print(s + "New best model.\n")
            self.least_wrong = curr_valid_wrong
            self.best_model = model
            self.patience_counter = 0
            torch.save(model, self.path_model)

        else:
            print(s + "Reset to previous best model.\n")
            model = self.best_model
            self.patience_counter += 1

        if self.patience_counter == self.patience * self.validation_amount:
            print(f"Model didn't improve for {self.patience} validation steps "
                  f"therefore training is terminated.")
            save_errors(self.path_errors, epoch, epoch_metrics["wrong"], self.least_wrong)
            data.store_parameters(self.path_parameters)
            sys.exit()

        if counter == self.validation_amount:
            save_errors(self.path_errors, epoch, epoch_metrics["wrong"], self.least_wrong)

        return counter, model

    def do_step(self, model, sample, data, loss_func, optimizer):

        model.train() if optimizer else model.eval()
        words, constituents = sample
        suffix, prefix = data.words2charIDvec(words, self.device)
        targets = get_targets(words, constituents, data, self.device)
        logits = model(prefix, suffix)
        loss = loss_func(logits, targets)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        correct = list(torch.argmax(logits, dim=1) == targets).count(True)
        wrong = len(targets) - correct
        return Counter({"correct": correct, "wrong": wrong, "loss": float(loss)})
