"""
P7 Tools - Aufgabe 8
Sentiment Prediction using LSTM

Gruppe:
Marcel Braasch
Nadja Seeberg
Sinem Kühlewind (geb. Demiraslan)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class LSTM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=config["num_embeddings"],
                                      embedding_dim=config["embedding_dim"])
        self.lstm = nn.LSTM(input_size=config["embedding_dim"],
                            hidden_size=config["hidden_dim"],
                            num_layers=config["num_layers"],
                            batch_first=True,
                            bidirectional=False)
        self.linear = nn.Linear(in_features=2*config["hidden_dim"],
                                out_features=config["num_classes"])
        self.dropout = nn.Dropout(p=config["dropout"])

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

class TextDataset(Dataset):

    def __init__(self, dir):
        self.dir = dir
        self.data = pd.read_csv(dir, sep="\t", header=None)

    def __getitem__(self, idx):
        label, text = self.data.iloc[idx]
        return label, map_seq_to_ids(text)

    def __len__(self):
        return len(self.data)

    def map_seq_to_ids(self, text):
        # TODO: Hier ensteht das Wort -> ID mapping.
        return text

class Trainer:

    def do_step(self, model, inputs, targets, optimizer=None):
        device = next(model.parameters()).device
        targets = targets.to(device=device)
        inputs = inputs.to(device=device)
        # TODO: Bis hierhin läuft das Programm.
        #       Inputs sind aktuell noch die rohen Stringsequenzen. Tn TextDataset müssen
        #       die einzelnen Sequenzen auf word_ids gemappet werden. Hier kann man auch
        #       schonmal die k häufigsten Wörter rausfinden und alle Wörter die nicht darin
        #       sind auf ein UNK Token (z.B. ID 1) mappen. Zudem müssen Sequenzen eine gemeinsame
        #       Länge haben damit sich im Batch verarbeitet werden können (z.B. max rausfinden
        #       und alle Sequenzen < max mit 0 padden)
        logits = model(inputs)
        #pad_mask = inputs > 0
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(logits, targets) #[pad_mask], targets[pad_mask])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss

    def do_epoch(self, model, dataloader, optimizer=None):
        model.eval() if optimizer is None else model.train()
        loss, acc = 0.0, 0.0
        for label, text in dataloader:
            loss += self.do_step(model, text, label, optimizer)
        return loss / len(dataloader.dataset)

    def train(self, model, dataloaders, config):
        best_model, best_epoch, best_acc = None, 0, float("-inf")
        optimizer = config["optimizer"](params=model.parameters(),lr=config["lr"],)
        for epoch in range(config["epochs"]):
            train_loss, train_acc = self.do_epoch(model, dataloaders["train"], optimizer)
            val_loss, val_acc = self.do_epoch(model, dataloaders["val"])
            if val_acc > best_acc:
                best_model, best_epoch, best_acc = model, epoch, val_acc
            if epoch - best_epoch > patience: break
        return best_model, best_epoch, best_acc


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
              "batch_size": 64}

    names = ["train", "dev", "test"]
    paths = ["/home/marcelbraasch/PycharmProjects/Tools/Aufgabe 7/data/sentiment.train.tsv",
             "/home/marcelbraasch/PycharmProjects/Tools/Aufgabe 7/data/sentiment.dev.tsv",
             "/home/marcelbraasch/PycharmProjects/Tools/Aufgabe 7/data/sentiment.test.tsv"]

    # Getting the data ready
    dirs = {name: path for name, path in zip(names, paths)}
    datasets = {name: TextDataset(dirs[name]) for name in names}
    dataloaders = {name: DataLoader(datasets[name],
                                    batch_size=config["batch_size"]) for name in names}

    # Init model and train
    model = LSTM(config=config)
    #if torch.cuda.is_available():
    #    model = model.to(device='cuda')
    trainer = Trainer()
    trainer.train(model, dataloaders, config)





