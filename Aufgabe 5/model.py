from collections import defaultdict
import sys
import os
from features import (
    avg_word_length_of,
    amount_exclamation_mark_of,
    length_of
    )
import math

def add(list_1, list_2):
    return [x + list_2[i] for i, x in enumerate(list_1)]

def sub(list_1, list_2):
    return [x - list_2[i] for i, x in enumerate(list_1)]

def mul(list_1, list_2):
    return [x * list_2[i] for i, x in enumerate(list_1)]

def div(list_1, list_2):
    return [x / list_2[i] for i, x in enumerate(list_1)]

def dot(list_1, list_2):
    return sum(mul(list_1, list_2))


class LogLinear:

    def __init__(self, mode):

        # Set parameters
        self.mode = mode
        self.paramfile = sys.argv[1]
        self.data_dir = sys.argv[2]
        if mode == "train":
            self.data_dir, self.paramfile = self.paramfile, self.data_dir
        self.classes = next(os.walk(self.data_dir))[1]

        # Parameters and feature functions
        self.features_functions = [avg_word_length_of,
                                   amount_exclamation_mark_of,
                                   length_of]
        self.theta = [1] * len(self.features_functions)

        # Load data once
        self.data = self.get_data()

    def predict(self):
        pass

    def fit(self):
        """Estimates probabilities given the frequencies. Then apply backoff smoothing."""

        if self.mode == "test": pass

        elif self.mode == "train":

            grad = [0] * len(self.theta)
            for sample, true_class in self.data:
                #true_score = self.feature_vec(true_class, sample)  # Left term in slides

                weighted_score = lambda sample, _class: math.exp(dot(self.feature_vec(sample, _class), self.theta))
                true_score = self.feature_vec(sample, true_class)
                Z = [weighted_score(sample, _class) for _class in self.classes]

                unnormalized_prob = weighted_score(sample, true_class)
                prob_class_given_doc = div(unnormalized_prob/Z)  # Right term in slides
                grad = add(grad, sub(true_score, prob_class_given_doc))
                s = 0

    def feature_vec(self, _class, sample):
        return [ff(sample[i], _class) for i, ff in enumerate(self.features_functions)]

    """
    def extract_features(self, files):
        data = {_class: [] for _class in self.classes}
        for file, _class in files:
            with open(f"train/{_class}/{file}", encoding="latin-1") as f:
                email = f.read()
                score = self.score(email, _class)
        return data
    
    def score(self, email, _class):
        sample = [amount_exclamation_mark_of(email, _class),
                  length_of(email, _class),
                  avg_word_length_of(email, _class)]
        return sum([feature_i * self.theta[i] for i, feature_i in enumerate(sample)])
    """

    def get_data(self):
        """
        Generator object that returns the next file and its
        corresponding class every time next() is called.
        """
        data = []
        for root, dirs, files in os.walk(self.data_dir):
            for _class in self.classes:
                if root.endswith(_class):
                    for file in files:
                        with open(f"train/{_class}/{file}", encoding="latin-1") as f:
                            data.append((f.read(), _class))
        return data

    def create_class_defaultdict(self, data_type):
        """Helper method to create defaultdict of classes."""
        return {_class: defaultdict(data_type) for _class in self.classes}