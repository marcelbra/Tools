"""
P7 Tools - Aufgabe 8
Sentiment Prediction using LSTM

Gruppe:
Marcel Braasch
Nadja Seeberg
Sinem Kühlewind (geb. Demiraslan)
"""

from utils import get_words_map_and_max_seq_len

import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd


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
                            bidirectional=False)
        self.linear = nn.Linear(in_features=config["hidden_dim"],
                                out_features=config["num_classes"])
        self.dropout = nn.Dropout(p=config["dropout"])

    def forward(self, inputs):
        inputs = inputs.to(torch.int64)
        x = self.embedding(inputs)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.linear(x)
        #try:
        # Select only the last layer and reshape
        batch_size = list(x.shape)[0]  # Can happen that batch is not full
        x = x[:,-1:,:].view(batch_size,
                            self.config["num_classes"])
        #except:
        #    print("x", x)
        #    print("x.shape", x.shape)
        #    print("Exiting programm.")
        #    sys.exit()

        return x

class TextDataset(Dataset):

    def __init__(self, path, words_to_ids, max_seq_len):
        self.path = path
        self.data = pd.read_csv(path, sep="\t", header=None)
        self.word_to_id = words_to_ids
        self.max_seq_len = max_seq_len

    def __getitem__(self, idx):
        label, text = self.data.iloc[idx]
        text = [self.word_to_id[x] for x in text.split()]
        text += [0] * (self.max_seq_len - len(text)) # Pad with 0s
        return label, torch.Tensor(text)

    def __len__(self):
        return len(self.data)

class Trainer:

    def do_step(self, model, inputs, targets, optimizer=None):
        device = next(model.parameters()).device
        targets, inputs = targets.to(device=device), inputs.to(device=device)
        logits = model(inputs)
        # Calc loss
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(logits, targets)
        if optimizer:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Calc acc
        preds = torch.argmax(logits, dim=1)
        correct = [x==y for x,y in zip(preds, targets)].count(1)
        acc = correct / list(preds.shape)[0]  # Normalize by batchsize
        return loss, acc

    def do_epoch(self, model, dataloader, optimizer=None):
        model.eval() if optimizer is None else model.train()
        n = len(dataloader.dataset)
        losses, accs = 0.0, 0.0
        for label, text in tqdm(dataloader):
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
            print(f"Starting epoch {epoch} / {config['epochs']}.")
            train_loss, train_acc = self.do_epoch(model, dataloaders["train"], optimizer)
            print(f"Validating epoch {epoch} / {config['epochs']}.")
            val_loss, val_acc = self.do_epoch(model, dataloaders["dev"])
            self.log(train_loss, train_acc, val_loss, val_acc,
                     train_losses, val_losses, train_accs, val_accs)
            if val_acc > best_acc:
                best_model, best_epoch, best_acc = model, epoch, val_acc
            if epoch - best_epoch > config["patience"]: break
        return best_model, best_epoch, best_acc

    def log(self, train_loss, train_acc, val_loss, val_acc,
            train_losses, val_losses, train_accs, val_accs):
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"Train loss: {train_loss}")
        print(f"Training acc: {train_acc}")
        print(f"Validation loss: {val_loss}")
        print(f"Validation acc: {val_acc}")
        print("\n"*3)

if __name__ == '__main__':

    config = {"num_embeddings": 5000,
              "num_layers": 2,
              "num_classes": 5,
              "embedding_dim": 300,
              "hidden_dim": 100,
              "dropout": 0.1,
              "optimizer": optim.AdamW,
              "epochs": 20,
              "patience": 20,
              "lr": 1e-3,
              "batch_size": 64,
              "vocab_size": 5000,}

    names = ["train", "dev", "test"]
    paths = ["/home/marcelbraasch/PycharmProjects/Tools/Aufgabe 7/data/sentiment.train.tsv",
             "/home/marcelbraasch/PycharmProjects/Tools/Aufgabe 7/data/sentiment.dev.tsv",
             "/home/marcelbraasch/PycharmProjects/Tools/Aufgabe 7/data/sentiment.test.tsv"]

    # Getting the data ready
    dirs = {name: path for name, path in zip(names, paths)}
    words_to_ids, max_seq_len = get_words_map_and_max_seq_len(dirs, k=config["vocab_size"])
    datasets = {name: TextDataset(dirs[name],
                                  words_to_ids,
                                  max_seq_len) for name in names}
    dataloaders = {name: DataLoader(datasets[name], batch_size=config["batch_size"]) for name in names}

    # Init model and train
    model = LSTM(config=config)
    #if torch.cuda.is_available():
    #    model = model.to(device='cuda')
    trainer = Trainer()
    trainer.train(model, dataloaders, config)





