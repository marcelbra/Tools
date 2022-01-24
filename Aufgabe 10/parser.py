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


class WordEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=config["num_prefixes"],
                                            embedding_dim=config["embeddings_dim"])
        self.embedding_layer = nn.Embedding(num_embeddings=config["num_suffixes"],
                                            embedding_dim=config["embeddings_dim"])
        self.forward_lstm = nn.LSTM(batch_first=True,
                                    input_size=config["embeddings_dim"],
                                    hidden_size=config["word_encoder_hidden_dim"],
                                    dropout=config["word_encoder_lstm_dropout"])
        self.backward_lstm = nn.LSTM(batch_first=True,
                                     input_size=config["embeddings_dim"],
                                     hidden_size=config["word_encoder_hidden_dim"],
                                     dropout=config["word_encoder_lstm_dropout"])

    def forward(self, prefixes, suffixes):
        prefixes = prefixes.to(torch.int64)
        suffixes = suffixes.to(torch.int64)
        pref = self.embedding_layer(prefixes)
        prefix_repr, _ = self.forward_lstm(pref)
        suf = self.embedding_layer(suffixes)
        suffix_repr, _ = self.backward_lstm(suf)
        word_representation = torch.cat((suffix_repr, prefix_repr), dim=2)
        return word_representation

class SpanEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bi_lstm = LSTM(batch_first=True,
                            bidirectional=True,
                            input_size=config["word_encoder_hidden_dim"]*2, # Because we concatenated
                            hidden_size=config["span_encoder_hidden_dim"],
                            dropout=config["span_encoder_lstm_dropout"])

    def forward(self, word_repr):
        batch_size, length, dim = word_repr.size()
        zeros = torch.zeros((batch_size,dim)).view(batch_size, 1, dim)
        padded_word_repr = torch.cat((zeros, word_repr, zeros), dim=1)
        spans, _ = self.bi_lstm(padded_word_repr)
        forward_repr, backward_repr = spans[:,:,:dim], spans[:,:,dim:]
        r_iks = torch.empty((batch_size, 0, 2*dim))
        for i in range(1, length+1):
            forward = forward_repr[:,i:,:] - forward_repr[:,:-i,:]
            backward = backward_repr[:,:-i,:] - backward_repr[:,i:,:]
            span_repr = torch.cat((forward, backward), dim=2)
            r_iks = torch.cat((r_iks, span_repr), dim=1)
        return r_iks

class Parser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(config["span_encoder_hidden_dim"]*2,
                      config["fc_hidden_dim"]),
            nn.Dropout(config["fc_dropout"]),
            nn.ReLU(),
            nn.Linear(config["fc_hidden_dim"],
                      config["num_class"])
        )

    def forward(self, span_representations):
        span_label_scores = self.feedforward(span_representations)
        batch_size, no_constiutuents, _ = span_label_scores.size()
        zeros = torch.zeros((batch_size,no_constiutuents, 1))
        span_label_scores = torch.cat((span_label_scores, zeros), dim=2) # Adding 0 vektor for "no constituent" class
        return span_label_scores

def main():

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
              }

    test_sequence = "This is a wonderful test sentence which will be processed by the LSTM ."
    prefix_ids = torch.Tensor([list(range(len(test_sequence.split())))
                               for _ in range(config["batch_size"])])
    suffix_ids = torch.Tensor([list(range(len(test_sequence.split())))[::-1]
                               for _ in range(config["batch_size"])])

    word_encoder = WordEncoder(config)
    word_repr = word_encoder(prefix_ids, suffix_ids)
    span_encoder = SpanEncoder(config)
    span_repr = span_encoder(word_repr)
    parser = Parser(config)
    scores = parser(span_repr)

if __name__=="__main__":
    main()