"""
P7 Tools - Aufgabe 11

Group:
Marcel Braasch
Nadja Seeberg
Sinem Kühlewind
"""

from trainer import Trainer
from Data import Data
from parser import Parser
from utils import load_data
from torch import optim
from tqdm import tqdm
import os

def main():

    paths = {"path_train": "./PennTreebank/train.txt",
             "path_dev": "./PennTreebank/dev.txt",
             "path_test": "./PennTreebank/test.txt.txt",
             "path_model": "./Outputs/model.pt",
             "path_data": "./Outputs/data.pkl",
             "path_parameters": "./Outputs/parameters.pkl",
             "path_errors": "./Outputs/errors.txt"}

    path_output = "./Outputs"
    if not os.path.exists(path_output):
        os.mkdir(path_output)

    data = load_data(paths=paths)

    model_config = {"num_chars": len(data.char2ID),
                    "num_class": len(data.label2ID),
                    "embeddings_dim": 100,
                    "word_encoder_hidden_dim": 100,
                    "span_encoder_hidden_dim": 250,
                    "fc_hidden_dim": 250,
                    "dropout": 0.4,
                    "span_encoder_num_layers": 2}

    optimizer_config = {"optimizer": optim.Adam}

    model = Parser(config=model_config)
    trainer = Trainer(paths=paths, epochs=50, config=optimizer_config)#, debug=True)
    trainer.train(model=model, data=data)

if __name__=="__main__":
    main()


#