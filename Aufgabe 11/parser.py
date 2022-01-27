"""
P7 Tools - Aufgabe 10

Group:
Marcel Braasch
Nadja Seeberg
Sinem Kühlewind
"""

import argparse
import torch
import torch.nn as nn
from torch.nn import LSTM
from data import Data


class WordEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(p=config["dropout"])
        self.embedding_layer = nn.Embedding(num_embeddings=config["num_chars"],
                                            embedding_dim=config["embeddings_dim"])

        self.forward_lstm = nn.LSTM(batch_first=True,
                                    input_size=config["embeddings_dim"],
                                    hidden_size=config["word_encoder_hidden_dim"])
        self.backward_lstm = nn.LSTM(batch_first=True,
                                     input_size=config["embeddings_dim"],
                                     hidden_size=config["word_encoder_hidden_dim"])

    def forward(self, prefixes, suffixes):
        #TODO: Richtiges Format übergeben
        prefixes = prefixes.to(torch.int64)
        suffixes = suffixes.to(torch.int64)
        pref = self.embedding_layer(prefixes)
        pref = self.dropout(pref)
        prefix_repr, _ = self.forward_lstm(pref)
        suf = self.embedding_layer(suffixes)
        suf = self.dropout(suf)
        suffix_repr, _ = self.backward_lstm(suf)
        word_representation = torch.cat((suffix_repr[:, -1],
                                         prefix_repr[:, -1]), dim=2)
        return word_representation

class SpanEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bi_lstm = LSTM(batch_first=True,
                            bidirectional=True,
                            input_size=config["word_encoder_hidden_dim"]*2, # Because we concatenated
                            hidden_size=config["span_encoder_hidden_dim"])

    def forward(self, word_repr):
        sent_length, repr_size = word_repr.size()
        zeros = torch.zeros((1, repr_size))
        padded_word_repr = torch.cat((zeros, word_repr, zeros),
                                     dim=1).unsqueeze(0)
        spans, _ = self.bi_lstm(padded_word_repr)
        forward_repr, backward_repr = spans.squeeze(0).split(dim, 1)

        forward = forward_repr[:-1]  # letzte Repräsentation entfernen
        backward = backward_repr[1:]  # erste Repräsentation entfernen

        span_reprs = [torch.cat((forward[l:] - forward[:-l],
                             backward[:-l] - backward[l]), -1)
                  for l in range(1, sent_length)]
        span_reprs = torch.cat(span_reprs)
        return span_reprs

class Parser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_encoder = WordEncoder(config)
        self.span_encoder = SpanEncoder(config)
        self.feedforward = nn.Sequential(
            nn.Linear(config["span_encoder_hidden_dim"]*2,
                      config["fc_hidden_dim"]),
            nn.Dropout(config["fc_dropout"]),
            nn.ReLU(),
            nn.Linear(config["fc_hidden_dim"],
                      config["num_class"])
        )

    def forward(self, prefix_ids, suffix_ids):
        word_repr = self.word_encoder(prefix_ids, suffix_ids)
        span_repr = self.span_encoder(word_repr)
        span_label_scores = self.feedforward(span_repr)
        batch_size, no_constiutuents, _ = span_label_scores.size()
        zeros = torch.zeros((batch_size,no_constiutuents, 1))
        span_label_scores = torch.cat((span_label_scores, zeros), dim=2) # Adding 0 vektor for "no constituent" class
        return span_label_scores

config = {"num_chars": 500,
          "num_class": 10,
          "embeddings_dim": 100,
          "word_encoder_hidden_dim": 100,
          "span_encoder_hidden_dim": 200,
          "word_encoder_lstm_dropout": 0.1,
          "span_encoder_lstm_dropout": 0.1,
          "fc_dropout": 0.1,
          "fc_hidden_dim": 32,
          "batch_size": 32,
          "lr": 1e-3,
          "dropout": 0.1
          }

parser = Parser(config)
data = Data("/Users/nadja/PycharmProjects/Tools/Aufgabe 11/data/test.txt",
            "/Users/nadja/PycharmProjects/Tools/Aufgabe 11/data/dev.txt")

sample = data.train_parses[0]
suffix_tensor, prefix_tensor = data.words2charIDvec(sample[0])
suffix_tensor = torch.Tensor(suffix_tensor)
prefix_tensor = torch.Tensor(prefix_tensor)
parser(suffix_tensor, prefix_tensor)


