"""


"""
import torch

from parser import Parser
from Data import Data
from utils import load_config, get_sentences_from_test, load_from_pickle, load_data, build_labels_vector


class Analyzer:

    def __init__(self, config):

        self.config = config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Load model
        model_config = load_config(config["model_config_path"])
        model_path = config["model_path"]
        self.model = torch.load(model_path, map_location='cpu')
        self.model.to(self.device)
        self.model.eval()

        # Load data
        self.test_data = get_sentences_from_test(config["test_file_path"])
        self.data_class = load_data(config["data_path"])

    def parse(self):

        sample = self.test_data[3]
        suffix, prefix = self.data_class.words2charIDvec(sample, self.device)
        logits = self.model(prefix, suffix)
        argmax_label_ids = torch.argmax(logits, dim=1)
        best_labels = [self.data_class.ID2label[i] for i in argmax_label_ids]
        best_scores = torch.max(logits, dim=1)[0]
        n = len(sample)
        # max_index = len(best_labels)
        # split = [0] * max_index
        labels_vector = build_labels_vector(n)
        best_indices = None

        for l in range(1, n):
            for i in range(n - l + 1):
                if l > 1:

                    # Compute argmax best split
                    best_score = float("-inf")
                    for j in range(i + 1, n):
                        for k in range(j + 1, n + 1):
                            ij_index = labels_vector.index(j - i) + i
                            jk_index = labels_vector.index(k - j) + j
                            ij_score = float(best_scores[ij_index])
                            jk_score = float(best_scores[jk_index])
                            new_score = ij_score + jk_score
                            if new_score > best_score:
                                best_score = new_score
                                best_indices = i, k
        self.output(best_indices[0], best_indices[1], best_labels, sample)

    def output(self, i, k, best_labels, test_sentence):
        sentences = test_sentence
        labels = [label for label in best_labels if label != '<unk>']
        print("(", labels[k-i])
        if k == i+1:
            print(test_sentence[i])
        else:
            self.output(i, k-i, labels, sentences)
            self.output(k-i, k, labels, sentences)
        print(")")


analyzer_config = {"model_config_path": "./Run-5/configs.txt",
                   "model_path": "./Run-5/model.pt",
                   "test_file_path": "./PennTreebank/test.txt",
                   "data_params_path": "./data_params.pkl",
                   "data_path": "./PennTreebank/data.pkl"}

analyzer = Analyzer(analyzer_config)
analyzer.parse()
