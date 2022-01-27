"""
P7 Tools - Aufgabe 11
Group:
Marcel Braasch
Nadja Seeberg
Sinem Kühlewind
"""
import pickle

from parser import Parser, WordEncoder, SpanEncoder
from Data import Data
from parser import Parser
import random
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class Trainer:

    def __init__(self, path_train, path_dev, path_test, config):
        self.path_train = path_train
        self.path_dev = path_dev
        self.path_test = path_test
        self.config = config
        self.train = True

    def load_data(self, path_train, path_dev):
        try:
            with open("data.pkl", "rb") as handle:
                return pickle.load(handle)
        except:
            return Data(path_train, path_dev)

    def do_train(self, model):
        #model.train() if optimizer else model.eval()
        optimizer = config["optimizer"](params=model.parameters(), lr=config["lr"])
        loss_func = nn.CrossEntropyLoss()
        epochs = self.config["epochs"]
        data = self.load_data(path_train, path_dev)

        for n in range(epochs):
            random.shuffle(data.train_parses)
            wrong, loss = self.do_epoch(model, loss_func, optimizer, data)

    def do_epoch(self, model, loss_func, optimizer, data):
        for sample in data.train_parses:
            wrong, loss = self.do_step(model, loss_func, optimizer, sample, data)

    def do_step(self, model, loss_func, optimizer, sample, data):
        words, constituents = sample
        suffix, prefix = data.words2charIDvec(words)
        suffix, prefix = torch.Tensor(suffix).to(torch.int64), torch.Tensor(prefix).to(torch.int64)
        targets = self.get_targets(words, constituents, data).type(torch.LongTensor)
        logits = model(prefix, suffix)
        loss = loss_func(logits, targets)
        if self.train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return 100, loss

    def get_targets(self, words, constituents, data):
        # Build labels vector
        label_vector = []
        for i in range(1, len(words) + 1):
            label_vector.extend([i] * (len(words) - i + 1))

        # Save index and label of respective constituent
        consts = []
        labels = []
        for constituent in constituents:
            label, start, end = constituent
            consts.append(label_vector.index(end - start) + start)
            labels.append(label)

        # Add label ID if it is a constituent else 0
        for i in range(len(label_vector)):
            if i in consts:
                label = labels[consts.index(i)]
                label_vector[i] = data.labelID(label)
            else:
                label_vector[i] = 0

        return torch.Tensor(label_vector)




config = {"num_chars": 1000,
          "num_class": 10,
          "embeddings_dim": 100,
          "word_encoder_hidden_dim": 100,
          "span_encoder_hidden_dim": 200,
          "word_encoder_lstm_dropout": 0.1,
          "span_encoder_lstm_dropout": 0.1,
          "fc_dropout": 0.1,
          "fc_hidden_dim": 32,
          "batch_size": 32,
          "optimizer": optim.SGD,
          "lr": 1e-3,
          "epochs": 50,
          "dropout": 0.2
          }

path_train = "../Aufgabe 08/PennTreebank/train.txt"
path_dev = "../Aufgabe 08/PennTreebank/dev.txt"
path_test = "../Aufgabe 08/PennTreebank/test.txt.txt"

model = Parser(config=config)

trainer = Trainer(path_train=path_train,
                  path_dev=path_dev,
                  path_test=path_test,
                  config=config)

trainer.do_train(model)

