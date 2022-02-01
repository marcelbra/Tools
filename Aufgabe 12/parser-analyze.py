"""


"""
import torch

from parser import Parser
from Data import Data
from utils import load_config, get_sentences_from_test, load_from_pickle, load_data

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
        label_scores = {}
        vscores = {}

        sample = self.test_data[0]
        suffix, prefix = self.data_class.words2charIDvec(sample, self.device)
        scores = self.model(prefix, suffix)
        print(scores.size())
        span_label_scores = torch.argmax(self.model(prefix, suffix), dim=1)
        n = len(sample)
        for l in range(1, n-1):
            for i in range(n-l):
                k = i + l
                label_scores[i, k] = span_label_scores[i]
                vscores[i, k] = scores[i, label_scores[i, k]] # stimmt das von der Dimension her?
                if l > 1:
                    pass

analyzer_config = {"model_config_path": "./Run-5/configs.txt",
                   "model_path": "./Run-5/model.pt",
                   "test_file_path": "./PennTreebank/test.txt",
                   "data_params_path": "./data_params.pkl",
                   "data_path": "./PennTreebank/data.pkl"}

analyzer = Analyzer(analyzer_config)
analyzer.parse()