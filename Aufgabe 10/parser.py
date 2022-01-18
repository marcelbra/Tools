"""
P7 Tools - Aufgabe 10
Neural Constituency Parsing (Network)

Gruppe:
Marcel Braasch
Nadja Seeberg
Sinem Kühlewind (geb. Demiraslan)
"""

import torch
import torch.nn as nn
from torch.nn import LSTM


class Parser(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate,
                 output_size):  # eigenes Config File/ Dict für jedes Netzwerk?
        super().__init__()
        self.linear_model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)

        )

    def forward(self, span_representations):
        span_label_scores = self.linear_model(span_representations)

        return span_label_scores

    class WordEncoder(nn.Module):
        def __init__(self, num_letters, emb_size, hidden_size, dropout_rate):  # Hidden_size is different for every net
            super().__init__()
            self.forward_lstm = LSTM()
            self.backward_lstm = LSTM()

        def forward(self, prefixes, suffixes):
            suffix_representation = self.forward_lstm(suffixes)
            prefix_representation = self.backward_lstm(prefixes)
            # concatenate representation tensors
            word_representations = torch.cat(prefix_representation, suffix_representation)

            return word_representations

    class SpanEncoder(nn.Module):
        def __init__(self, input_size, hidden_size, dropout_rate):
            super().__init__()
            self.bidirectional_lstm = LSTM(bidirectional=True)

        def forward(self, word_representations):

            #add vector of 0s to beginning and end of input matrix
            zero_vec = torch.zeros((word_representations.size(dim=1), word_representations.size(dim=0)))
            padded_input = torch.cat((zero_vec.unsqueeze(0), word_representations, zero_vec.unsqueeze(0)), dim=2)

            span_representations = self.bidirectional_lstm(padded_input)

            return span_representations


if __name__ == '__main__':
    parser = Parser()
