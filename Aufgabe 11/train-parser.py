"""
P7 Tools - Aufgabe 11
Group:
Marcel Braasch
Nadja Seeberg
Sinem Kühlewind
"""

from parser import Parser, WordEncoder, SpanEncoder
from Data import Data
import random
import torch.nn as nn
import torch
import torch.optim as optim


class Trainer:

    def __init__(self, path_train, path_dev, path_test, model):
        self.path_train = path_train
        self.path_dev = path_dev
        self.path_test = path_test
        self.data = Data(path_train, path_dev)
        self.model = model

    def train(self):
        num_epochs = 50

        for n in range(num_epochs):
            random.shuffle(self.data.train_parses)
            for sample in self.data:
                words = sample[0]
                suffix_tensor, prefix_tensor = self.data.words2charIDvec(words)
                word_encoder = WordEncoder(config)
                word_repr = word_encoder(prefix_ids, suffix_tensor)
                span_encoder = SpanEncoder(config)
                span_repr = span_encoder(word_repr)
                parser = Parser(config)
                scores = parser(span_repr)

    def loss(self, prefixes, suffixes, constituents):
        optimizer = config["optimizer"](params=model.parameters(), lr=config["lr"], )
        span_label_scores = Parser()
        loss_func = nn.CrossEntropyLoss()

        for sentence_constituent_pair in self.data.train_parses:
            sentence = sentence_constituent_pair[0]
            constituents = sentence_constituent_pair[1]
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

        return None


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
          "optimizer": "optim.SGD",
          "lr": "1e-3"
          }

path_train = "../Aufgabe 08/PennTreebank/train.txt"
path_dev = "../Aufgabe 08/PennTreebank/dev.txt"
path_test = "../Aufgabe 08/PennTreebank/test.txt.txt"

model = Parser(config=config)

trainer = Trainer(path_train=path_train,
                  path_dev=path_dev,
                  path_test=path_test)

test_sequence = "This is a wonderful test sentence which will be processed by the LSTM ."
prefix_ids = torch.Tensor([list(range(len(test_sequence.split())))
                           for _ in range(config["batch_size"])])
suffix_ids = torch.Tensor([list(range(len(test_sequence.split())))[::-1]
                           for _ in range(config["batch_size"])])


#class FullParser(WordEncoder, SpanEncoder, Parser):


