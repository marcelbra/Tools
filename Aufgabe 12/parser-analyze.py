"""
P7 EET - Aufgabe 12
Parser: Anwendung

Group:
Marcel Braasch
Nadja Seeberg
Sinem Kühlewind
"""

import torch
import sys
from parser import Parser
from Data import Data
from utils import load_config, get_test_data, load_data, Index

class Printer:

    def __init__(self, labels, splits, index, sentence):
        self.labels = labels
        self.splits = splits
        self.index = index
        self.sentence = sentence

    def build_parse(self, i, k):
        label = self.labels[self.index(i, k)]
        number_closing = 0
        if label != "<unk>":  # Omit unknown consituents
            number_closing += 1
            if len(label.split()) == 2:  # Resolve merged consituents
                label = label.split()[0] + "(" + label.split()[1]
                number_closing += 1
            print(f"({label}", end="")
        if k==i+1:
            print(f" {self.sentence[i]}", end="")
        else:
            split_index = self.splits[self.index(i,k)]
            self.build_parse(i, split_index)
            self.build_parse(split_index, k)
        print(")" * number_closing, end="")

class Analyzer:

    def __init__(self, paths):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = torch.load(paths["model"], map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        self.test_data, self.trees = get_test_data(paths["test_file"])
        self.data_class = Data(paths["parameters"])

    def parse(self, sentences):

        if isinstance(sentences, str):
            sentences = [sentences.split()]
        else:
            sentences = self.test_data[:15]

        for sentence in sentences:
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
            printer.build_parse(0, n)
            print()

def main():
    paths = {"model": "./Data/model.pt",
             "test_file": "./Data/test.txt",
             "parameters": "./Data/parameters.pkl"}
    analyzer = Analyzer(paths)
    if len(sys.argv)==2:
        analyzer.parse(sys.argv[1])
    else:
        analyzer.parse(None)

main()