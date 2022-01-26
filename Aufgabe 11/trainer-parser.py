"""
P7 Tools - Aufgabe 11

Group:
Marcel Braasch
Nadja Seeberg
Sinem Kühlewind
"""

from parser import Parser
from Data import Data

class Trainer:

    def __init__(self, path_train, path_dev, path_test, model):
        
        self.path_train = path_train
        self.path_dev = path_dev
        self.path_test = path_test
        self.data = Data(path_train, path_dev)
        self.model = model

    def train(self):
        pass

path_train = "../Aufgabe 08/PennTreebank/train.txt"
path_dev = "../Aufgabe 08/PennTreebank/dev.txt"
path_test = "../Aufgabe 08/PennTreebank/test.txt.txt"

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

model = Parser(config=config)
trainer = Trainer(path_train=path_train,
                  path_dev=path_dev,
                  path_test=path_test)

        