"""


"""
import torch

from parser import Parser
from Data import Data
from utils import load_config, get_test_data, load_data, Index


class Printer:

    def __init__(self, labels, splits, index, sentence):

        self.labels = labels
        self.splits = splits
        self.index = index
        self.sentence = sentence

    def output(self, i, k):
        # if self.labels[self.index(i, k)] != "<unk>":
        print(f"({self.labels[self.index(i, k)]}", end="")
        if k==i+1:
            print(f" {self.sentence[i]}", end="")
        else:
            spit_index = self.splits[self.index(i,k)]
            self.output(i, spit_index)
            self.output(spit_index, k)
        print(")", end="")

class Analyzer:

    def __init__(self, config):

        # Load model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_config = load_config(config["model_config_path"])
        model_path = config["model_path"]
        self.model = torch.load(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

        # Load data
        self.test_data, self.trees = get_test_data(config["test_file_path"])
        self.data_class = load_data(config["data_path"])

    def parse(self, sentence):

        o = 0

        sentence = self.test_data[o]
        inputs = self.data_class.words2charIDvec(sentence, self.device)
        logits = self.model(*inputs)
        labels = [self.data_class.ID2label[i] for i in torch.argmax(logits, dim=1)]
        scores = torch.max(logits, dim=1)[0]
        splits = [""] * len(labels)
        n = len(sentence)
        index = Index(n)

        for l in range(2, n + 1):
            for i in range(n - l + 1):

                k = i + l
                all_splits = [float(scores[index(i, j)] + scores[index(j, k)]) for j in range(i + 1, k)]
                argmax_j = i + all_splits.index(max(all_splits)) + 1
                scores[index(i, k)] += scores[index(i, argmax_j)] + scores[index(argmax_j, k)]
                splits[index(i,k)] = argmax_j

        printer = Printer(labels, splits, index, sentence)
        printer.output(0, n)
        print(f"\n{self.trees[o]}")

analyzer_config = {"model_config_path": "./Run-5/configs.txt",
                   "model_path": "./Run-5/model.pt",
                   "test_file_path": "./PennTreebank/test.txt",
                   "data_params_path": "./data_params.pkl",
                   "data_path": "./PennTreebank/data.pkl"}

analyzer = Analyzer(analyzer_config)
analyzer.parse("Test sentence")
