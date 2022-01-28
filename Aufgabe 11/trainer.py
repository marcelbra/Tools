from parser import Parser, WordEncoder, SpanEncoder
from utils import get_targets, load_data, save_errors
from Data import Data

from collections import defaultdict, Counter
from tqdm import tqdm
import pickle
import random

import torch.nn as nn
import torch.optim as optim
import torch

class Trainer:

    def __init__(self, paths, epochs, config, debug=False):

        self.path_model = paths["path_model"]
        self.path_parameters = paths["path_parameters"]
        self.path_errors = paths["path_errors"]
        self.config = config
        self.epochs = epochs
        self.debug = debug
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def train(self, model, data):

        optimizer = self.config["optimizer"](params=model.parameters())
        loss_func = nn.CrossEntropyLoss()
        model.to(self.device)
        metrics = defaultdict(list)
        lowest_wrong_amount = float("inf")

        for epoch in range(self.epochs):

            print(f"Starting epoch: {epoch}")

            current_train_metrics = self.do_epoch(model, data, loss_func, optimizer)
            current_valid_metrics = self.do_epoch(model, data, loss_func)
            metrics["train"].append(current_train_metrics)
            metrics["valid"].append(current_valid_metrics)

            print(f"Train: {dict(current_train_metrics)}")
            print(f"Valid: {dict(current_valid_metrics)}")

            current_wrong_amount = current_valid_metrics["wrong"]
            if current_wrong_amount < lowest_wrong_amount:
                lowest_wrong_amount = current_wrong_amount
                torch.save(model, self.path_model)

            save_errors(self.path_errors, epoch, current_train_metrics, current_valid_metrics)

        data.store_parameters(self.path_parameters)

    def do_epoch(self, model, data, loss_func, optimizer=None):

        if self.debug:
            # Overfit on small sample size to confirm model has no bugs
            curr_dataset = data.train_parses
            curr_dataset = curr_dataset[:50]
        else:
            curr_dataset = data.train_parses if optimizer else data.dev_parses
            random.shuffle(curr_dataset)

        epoch_metrics = Counter()
        for sample in tqdm(curr_dataset, position=0, leave=True):
        # for sample in curr_dataset:
                epoch_metrics += self.do_step(model, sample, data, loss_func, optimizer)

        return epoch_metrics

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

#