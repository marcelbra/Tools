"""
P7 Experimente, Evaluierung, Tools
Exercise 10 - Parsing with neural networks

Group:
Marcel Braasch
Nadja Seeberg
Sinem Kühlewind
"""

import sys
import torch
import torch.nn as nn
from config import *


# first LSTM to encode words (word representations)
class WordEncoder(nn.Module):
    def __init__(self, num_letters, embedding_dim, hidden_size, dropout_rate):
        super().__init__()
        self.num_letters = num_letters
        # TODO-Frage: num_embeddings ist len(woerter_im_input_satz), oder?
        self.embedding_layer = nn.Embedding(num_embeddings=10,
                                            embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            bidirectional=False,
                            dropout=dropout_rate)

    def forward(self, prefixes, suffixes):
        prefixes = prefixes.to(torch.int64)
        suffixes = suffixes.to(torch.int64)
        pref = self.embedding_layer(prefixes)
        pref, _ = self.lstm(pref)
        suf = self.embedding_layer(suffixes)
        suf, _ = self.lstm(suf)
        word_representation = torch.cat(pref, suf)
        return word_representation


class SpanEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            bidirectional=False,
                            dropout=dropout_rate)

    def forward(self, word_representation):
        span_representation = self.lstm(word_representation)
        # TODO-Frage: wie werden die span representations berechnet??
        return span_representation


class Parser(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            bidirectional=False,
                            dropout=dropout_rate)

    def forward(self, span_representation):
        span_label_scores = self.lstm(span_representation)
        return span_label_scores



if __name__ == "__main__":
    word_encoder = WordEncoder(num_letters=config.Word_Encoder["num_letters"],
                               embedding_dim=config.Word_Encoder["embedding_dim"],
                               hidden_size=config.Word_Encoder["hidden_size"],
                               dropout_rate=config.Word_Encoder["dropout_rate"])
    # TODO: hier dann for word in sentence: Prefixes und Suffixes extrahieren und dem forward von word_encoder übergeben
    # dann dem SpanEncoder etc.

    span_encoder = SpanEncoder(input_size=config.Span_Encoder["input_size"],
                               hidden_size=config.Span_Encoder["hidden_size"],
                               dropout_rate=config.SpanEncoder["dropout_rate"])

    parser = Parser(input_size=config.Parser["input_size"],
                    hidden_size=config.Parser["hidden_size"],
                    dropout_rate=config.Parser["dropout_rate"])




