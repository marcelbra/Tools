"""
P7 Tools - Aufgabe 11

Group:
Marcel Braasch
Nadja Seeberg
Sinem Kühlewind
"""

from torch import optim
from tqdm import tqdm
import os
import random

from trainer import Trainer
from Data import Data
from parser import Parser
from utils import (load_data, save_configs,
                   create_directories, get_args)

def main():

    # Path and saving set-up
    path_output = create_directories()
    paths = {"path_train": "./PennTreebank/train.txt",
             "path_dev": "./PennTreebank/dev.txt",
             "path_test": "./PennTreebank/test.txt.txt",
             "path_data": "./PennTreebank/data.pkl",
             "path_model": path_output + "model.pt",
             "path_parameters": path_output + "parameters.pkl",
             "path_errors": path_output + "errors.txt",
             "path_configs": path_output + "configs.txt"}

    # Data set -up
    data = load_data(paths=paths)

    # Config & parameter set-up
    config = {"num_chars": len(data.char2ID), "num_class": len(data.label2ID)}
    config.update(vars(get_args()))
    save_configs(path_configs=paths["path_configs"], config=config)

    # Model & training set-up
    model = Parser(config=config)
    trainer = Trainer(paths=paths, epochs=50, config=config)
    print("\nStarting training run with following config:\n")
    print(*config.items(), sep="\n", end="\n")
    print(f"\nData and logs will be saved to '{path_output}'.\n")
    trainer.train(model=model, data=data)

if __name__=="__main__":
    main()

