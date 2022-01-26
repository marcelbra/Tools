from Data import Data
from Aufgabe\ 10 import Parser
import torch.nn as nn
import torch
import random


class TrainParser:
    def __init__(self):
        self.path_train = "./data/train.txt"
        self.path_dev = "./data/dev.txt"


    def train(self):
        num_epochs = 50
        data = Data(self.path_train, self.path_dev)

        for n in range(num_epochs):
            random.shuffle(data.train_parses)


    def loss(self, prefixes, suffixes, constituents):
        data = Data(self.path_train, self.path_dev)
        span_label_scores = Parser(prefixes, suffixes)
        loss_func = nn.CrossEntropyLoss()

        for sentence_constituent_pair in data.train_parses:
            sentence = sentence_constituent_pair[0]
            label_vector = []
            len_counter = len(sentence)
            for i in range(1, len(sentence)+1):
                label_vector.extend([i]*len_counter)
                len_counter -= 1
                for const in constituents:
                    span_length = const[2] - const[1]
                    vector_ix = label_vector.index(span_length) + const[1]
                    label_vector[vector_ix] = data.labelID(const[0])

                loss = loss_func(label_vector, span_label_scores) #input für loss function?

        return None

if __name__ == '__main__':
    train = TrainParser()

