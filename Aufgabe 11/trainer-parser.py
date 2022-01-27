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

    def load_data(self, path_train, path_dev):
        try:
            with open("data.pkl", "rb") as handle:
                return pickle.load(handle)
        except:
            return Data(path_train, path_dev)

    def train(self, model):
        optimizer = config["optimizer"](params=model.parameters(), lr=config["lr"])
        loss_func = nn.CrossEntropyLoss()
        epochs = self.config["epochs"]
        data = self.load_data(path_train, path_dev)
        for n in range(epochs):
            random.shuffle(data.train_parses)
            acc, loss = self.do_epoch(model, loss_func, optimizer, data)

    def do_epoch(self, model, loss, optimizer, data):
        for sample in data.train_parses:
            acc, loss = self.loss(model, loss, optimizer, sample, data)


    def loss(self, model, loss, optimizer, sample, data):

        words, consituents = sample
        suffix, prefix = data.words2charIDvec(words)
        logits = model(prefix, suffix)

        label_vector = []
        len_counter = len(sentence)
        for i in range(1, len(sentence) + 1):
            label_vector.extend([i] * len_counter)
            len_counter -= 1
            for const in constituents:
                span_length = const[2] - const[1]
                vector_ix = label_vector.index(span_length) + const[1]
                label_vector[vector_ix] = self.data.labelID(const[0])

                #TODO: Berechnung num_errors

            loss = loss_func(label_vector, span_label_scores) # input für loss function?
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    #
    #     return None


config = {"num_suffixes": 500,
          "num_prefixes": 500,
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
          "epochs": 50
          }

path_train = "../Aufgabe 08/PennTreebank/train.txt"
path_dev = "../Aufgabe 08/PennTreebank/dev.txt"
path_test = "../Aufgabe 08/PennTreebank/test.txt.txt"

model = Parser(config=config)

trainer = Trainer(path_train=path_train,
                  path_dev=path_dev,
                  path_test=path_test,
                  config=config)

trainer.train(model)

# test_sequence = "This is a wonderful test sentence which will be processed by the LSTM ."
# prefix_ids = torch.Tensor([list(range(len(test_sequence.split())))
#                            for _ in range(config["batch_size"])])
# suffix_ids = torch.Tensor([list(range(len(test_sequence.split())))[::-1]
#                            for _ in range(config["batch_size"])])


#class FullParser(WordEncoder, SpanEncoder, Parser):

