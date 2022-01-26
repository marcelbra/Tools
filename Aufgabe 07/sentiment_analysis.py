"""
P7 Tools - Aufgabe 8
Sentiment Prediction using LSTM

Gruppe:
Marcel Braasch
Nadja Seeberg
Sinem Kühlewind (geb. Demiraslan)
"""

from utils import get_words_map_and_max_seq_len, sample_configs
import matplotlib.pyplot as plt

import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from math import exp
from torch.utils.data import Dataset, DataLoader


#import pandas as pd


class LSTM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(num_embeddings=config["num_embeddings"] + 2,  # For special tokes
                                      embedding_dim=config["embedding_dim"])
        self.lstm = nn.LSTM(input_size=config["embedding_dim"],
                            hidden_size=config["hidden_dim"],
                            num_layers=config["num_layers"],
                            batch_first=True,
                            bidirectional=False,
                            dropout=config["dropout"])
        self.linear = nn.Linear(in_features=config["hidden_dim"],
                                out_features=config["num_classes"])
        self.dropout = nn.Dropout(p=config["dropout"])

    def forward(self, inputs):
        inputs = inputs.to(torch.int64)
        x = self.embedding(inputs)
        x, _ = self.lstm(x)
        self.dropout(x)
        x = self.linear(x[:,-1,:])
        return x

class TextDataset(Dataset):

    def __init__(self, path, words_to_ids, max_seq_len):
        self.path = path
        #self.data = pd.read_csv(path, sep="\t", header=None)
        self.word_to_id = words_to_ids
        self.max_seq_len = max_seq_len
        self.data = self.load_data(path)

    def __getitem__(self, idx):
        #label, text = self.data.iloc[idx]
        label, text = self.data[idx]
        text = [self.word_to_id[x] for x in text.split()]
        text += [0] * (self.max_seq_len - len(text)) # Pad with 0s
        return label, torch.Tensor(text)

    def __len__(self):
        return len(self.data)

    def load_data(self, path):
        data = []
        with open(path, "r") as f:
            for line in f:
                line = line.split("\t")
                data.append((int(line[0]), line[1][:-1]))
        return data

class Trainer:

    def do_step(self, model, inputs, targets, optimizer=None):
        device = next(model.parameters()).device
        targets, inputs = targets.to(device=device), inputs.to(device=device)
        logits = model(inputs)
        # Calc loss
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(logits, targets)
        if optimizer:
            #optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-5, momentum=0.9)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Calc acc
        preds = torch.argmax(logits, dim=1)
        correct = [x==y for x,y in zip(preds, targets)].count(1)
        acc = correct / list(preds.shape)[0]  # Normalize by batchsize
        return loss, acc

    def do_epoch(self, model, dataloader, optimizer=None):
        model.train() if optimizer else model.eval()
        losses, accs = 0.0, 0.0
        n = len(dataloader)
        for label, text in dataloader:
            loss, acc = self.do_step(model, text, label, optimizer)
            losses += loss
            accs += acc
        losses = losses / n
        accs = accs / n
        return losses, accs

    def train(self, model, dataloaders, config):
        best_model, best_epoch, best_acc = None, 0, float("-inf")
        optimizer = config["optimizer"](params=model.parameters(),lr=config["lr"],)
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        for epoch in range(config["epochs"]):
            train_loss, train_acc = self.do_epoch(model, dataloaders["train"], optimizer)
            val_loss, val_acc = self.do_epoch(model, dataloaders["dev"])
            self.log(train_loss, train_acc, val_loss, val_acc,
                     train_losses, val_losses, train_accs, val_accs, epoch)
            if val_acc > best_acc:
                best_model, best_epoch, best_acc = model, epoch, val_acc
            if epoch - best_epoch > config["patience"]:
                break
        return (best_model, best_epoch, best_acc,
                train_losses, val_losses,
                train_accs, val_accs)

    def log(self, train_loss, train_acc, val_loss, val_acc,
            train_losses, val_losses, train_accs, val_accs, epoch):
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        #print(f"Epoch {epoch} train acc: {train_acc}")
        #print(f"     Train loss: {train_loss}")
        #print(f"Validation loss: {val_loss}")
        #print(f"   Training acc: {train_acc}")
        #print(f" Validation acc: {val_acc}")

if __name__ == '__main__':

    """    
    32 Accuracy
    config = {"num_embeddings": 5000,
              "vocab_size": 5000,
              "num_classes": 5,
              "patience": 25,
              "dropout": 0.0,
              "num_layers": 1,
              "embedding_dim": 300,
              "hidden_dim": 300,
              "optimizer": optim.RMSprop,
              "epochs": 30,
              "lr": 5e-6,
              "batch_size": 100,
              }
    """

    config_ranges = {'embedding_dim': [2 ** x for x in range(6, 10)],
                     'hidden_dim': [2 ** x for x in range(6, 10)],
                     'batch_size': [2 ** x for x in range(4, 9)],
                     'dropout': [x / 10 for x in range(0, 6)],
                     'optimizer': [optim.AdamW, optim.Adam, optim.SGD, optim.RMSprop, optim.Adagrad, optim.Adadelta],
                     'lr': [exp(-x) for x in range(5, 8)] + [exp(-x / 2) for x in range(5, 8)] + [
                         eval(f"1e-{x}") for x in range(3, 7)],
                     }

    config = {"num_embeddings": 5000,
              "vocab_size": 5000,
              "num_classes": 5,
              "patience": 10,
              "num_layers": 2,
              "epochs": 30
              }

    names = ["train", "dev", "test"]
    paths = ["/home/marcelbraasch/PycharmProjects/Tools/Aufgabe 7/data/sentiment.train.tsv",
             "/home/marcelbraasch/PycharmProjects/Tools/Aufgabe 7/data/sentiment.dev.tsv",
             "/home/marcelbraasch/PycharmProjects/Tools/Aufgabe 7/data/sentiment.test.tsv"]

    # Getting the data ready
    dirs = {name: path for name, path in zip(names, paths)}
    words_to_ids, max_seq_len = get_words_map_and_max_seq_len(dirs, k=config["vocab_size"])
    datasets = {name: TextDataset(dirs[name], words_to_ids, max_seq_len) for name in names}

    best_acc = float("-inf")
    best_model = None
    best_config = None

    # Doing random hyperparameter search
    budget = 80
    configs = sample_configs(budget=budget, config_ranges=config_ranges)
    for sampled_config in tqdm(configs):

        new_config = dict(list(config.items()) + list(sampled_config.items()))
        dataloaders = {name: DataLoader(datasets[name], batch_size=new_config["batch_size"]) for name in names}

        # Init model
        model = LSTM(config=new_config)
        if torch.cuda.is_available():
            model = model.to(device='cuda')

        # Train
        trainer = Trainer()
        (best_current_model, best_epoch, best_current_acc,
         train_losses, val_losses, train_accs, val_accs) = trainer.train(model, dataloaders, new_config)
        if best_current_acc > best_acc:
            best_model = best_current_model
            best_config = new_config
            best_acc = best_current_acc

        # Plot results
        debug = False
        if debug:
            steps = list(range(len(train_losses)))
            train_losses = [float(x) for x in train_losses]
            val_losses = [float(x) for x in val_losses]
            train_accs = [float(x) for x in train_accs]
            val_accs = [float(x) for x in val_accs]
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].plot(steps, train_losses)
            axs[0, 0].set_title('Training Loss')
            axs[0, 1].plot(steps, val_losses, 'tab:orange')
            axs[0, 1].set_title('Validation Loss')
            axs[1, 0].plot(steps, train_accs, 'tab:green')
            axs[1, 0].set_title('Training Accuracy')
            axs[1, 1].plot(steps, val_accs, 'tab:red')
            axs[1, 1].set_title('Validation Accuracy')
            fig.show()

    print(f"Best model reached accuracy of {best_acc}.")
    print(f"Its config was: {best_config}.")

    #_, acc = trainer.do_epoch(best_model, dataloader["train"])
    #print(acc)