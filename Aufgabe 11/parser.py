"""
P7 Tools - Aufgabe 10

Group:
Marcel Braasch
Nadja Seeberg
Sinem Kühlewind
"""

import argparse
import pickle
import torch
import torch.nn as nn
from torch.nn import LSTM
from Data import Data

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class WordEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=config["num_chars"],
                                            embedding_dim=config["embeddings_dim"])
        self.dropout = nn.Dropout(p=config["dropout"])
        self.forward_lstm = nn.LSTM(batch_first=True,
                                    input_size=config["embeddings_dim"],
                                    hidden_size=config["word_encoder_hidden_dim"])
        self.backward_lstm = nn.LSTM(batch_first=True,
                                     input_size=config["embeddings_dim"],
                                     hidden_size=config["word_encoder_hidden_dim"])

    def forward(self, prefixes, suffixes):
        prefixes = self.embedding_layer(prefixes)
        suffixes = self.embedding_layer(suffixes)
        prefixes = self.dropout(prefixes)
        suffixes = self.dropout(suffixes)
        prefix_repr, _ = self.forward_lstm(prefixes)
        suffix_repr, _ = self.backward_lstm(suffixes)
        word_representation = torch.cat((suffix_repr[:,-1,:], prefix_repr[:,-1,:]), dim=1)
        word_representation = self.dropout(word_representation)
        return word_representation

class SpanEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bi_lstm = LSTM(batch_first=True,
                            bidirectional=True,
                            num_layers=config["span_encoder_num_layers"],
                            input_size=config["word_encoder_hidden_dim"]*2, # Because we concatenated
                            hidden_size=config["span_encoder_hidden_dim"],
                            dropout=config["dropout"])
        self.dropout = nn.Dropout(p=config["dropout"])


    def forward(self, word_representation):
        sent_length, repr_size = word_representation.size()
        zeros = torch.zeros((1, repr_size)).to(device)
        padded_word_repr = torch.cat((zeros, word_representation, zeros), dim=0).unsqueeze(0)
        padded_word_repr = self.dropout(padded_word_repr)
        spans, _ = self.bi_lstm(padded_word_repr)
        forward, backward = torch.split(spans.squeeze(0), int(spans.size()[-1] / 2), 1)
        forward, backward = forward[:-1], backward[1:]
        span_reprs = [torch.cat((forward[l:] - forward[:-l],
                                 backward[:-l] - backward[l:]), -1)
                      for l in range(1, sent_length + 1)]
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
            nn.Dropout(config["dropout"]),
            nn.ReLU(),
            nn.Linear(config["fc_hidden_dim"],
                      config["num_class"])
        )

    def forward(self, prefix_ids, suffix_ids):
        word_repr = self.word_encoder(prefix_ids, suffix_ids)
        span_repr = self.span_encoder(word_repr)
        span_label_scores = self.feedforward(span_repr)
        return span_label_scores

