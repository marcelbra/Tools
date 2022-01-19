"""
P7 Tools - Aufgabe 10

Group:
Marcel Braasch
Nadja Seeberg
Sinem Kühlewind
"""

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
        # return prefix_repr, suffix_repr

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

        r_iks = torch.empty((length,length,batch_size,1,dim*2))

        def create_r_ik(forward_repr, backward_repr, i, k):
            # 1. Dim = Batch, 2. Dim = Word, 3. Dim = Embedding
            forward_i = forward_repr[:,i:i+1,:]
            backward_i = backward_repr[:,i+1:i+ 2,:]
            forward_k = forward_repr[:,k:k+1,:]
            backward_k = backward_repr[:,k+1:k+ 2,:]
            return torch.cat((forward_k - forward_i, backward_i-backward_k),dim=2)

        for i in range(1, length-1):
            for k in range(i+1, length-1):
                r_iks[i][k] = create_r_ik(forward_repr, backward_repr, i, k)

        r_iks = r_iks.view(length,length,batch_size,dim*2)
        return r_iks

class Parser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(config["span_encoder_hidden_dim"]*2,
                      config["fc_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["fc_hidden_dim"],
                      config["num_class"])
        )

    def forward(self, span_representations):
        span_label_scores = self.feedforward(span_representations)
        return span_label_scores


def main():
    config = {"num_suffixes": 500,
              "num_prefixes": 500,
              "embeddings_dim": 100,
              "word_encoder_hidden_dim": 100,
              "span_encoder_hidden_dim": 200,
              "word_encoder_lstm_dropout": 0.1,
              "span_encoder_lstm_dropout": 0.1,
              "batch_size": 32,
              "fc_hidden_dim": 32,
              "num_class": 10
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