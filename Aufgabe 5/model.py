"""
P7 Experimente, Evaluierung und Tools
Aufgabe 5 - Spam classification using log linear models

Group:
Marcel Braasch
Nadja Seeberg
Sinem Kühlewind (geb. Demiraslan)
"""

from collections import defaultdict
import sys
import os
from features import (
    avg_word_length_pos,
    avg_word_length_neg,
    amount_exclamation_mark_pos,
    amount_exclamation_mark_neg,
    length_pos,
    length_neg
    )
import math
from vector_operations import add, sub, mul, div, dot, create_vec


class LogLinear:

    def __init__(self, mode, paramfile, data_dir):

        # Set parameters
        self.mode = mode
        self.paramfile = paramfile
        self.data_dir = data_dir
        if mode == "train":
            self.data_dir, self.paramfile = self.paramfile, self.data_dir
        self.classes = ["ham", "spam"] #TODO: next(os.walk(self.data_dir))[1] # irgendwas spinnt hier bei mir

        # Parameters and feature functions
        self.features_functions = [avg_word_length_pos,
                                   avg_word_length_neg,
                                   length_neg,
                                   length_pos,
                                   amount_exclamation_mark_pos,
                                   amount_exclamation_mark_neg]
        self.n = len(self.features_functions)
        self.theta = [1] * self.n
        self.eta = 1e-3

        # Load data once
        self.data = self.get_data()

    def predict(self):
        pass

    def fit(self):
        """Estimates probabilities given the frequencies. Then apply backoff smoothing."""

        if self.mode == "test": pass

        elif self.mode == "train":
            epochs = range(20)
            grad = create_vec(0, self.n)
            for epoch in epochs:
                scores = create_vec(0, self.n)
                for sample, true_class in self.data:
                    # Feature vector, left term in slides
                    feature_score_left = self.feature_vec(sample, true_class)
                    # Expectation times feature vector, right term in slides
                    prob_class_given_doc = 0
                    counter = 0  # Counts at which vector entry we are right now
                    for _class in self.classes:
                        # This is the part right of the expectation (see slides)
                        feature_score_right = self.features_functions[counter](sample, _class)
                        counter += 1
                        # This is the normalization
                        Z = sum([math.exp(dot(self.theta, self.feature_vec(c, sample))) for c in self.classes])
                        # Unnormalized score
                        class_prob = math.exp(dot(self.theta, self.feature_vec(_class, sample)))
                        # Combine all terms
                        prob_class_given_doc += (class_prob * feature_score_right) / Z
                    # Combine left and right term
                    score = sub(feature_score, expectation_times_feature)
                    # Accumulate gradient
                    scores = add(scores, score)
                # Update gradient
                grad = add(grad, scores)
                # Update parameters
                self.theta = add(self.theta, mul(create_vec(self.eta, n), grad))

                # TODO: Evaluation / logging after every gradient update
                # TODO: Weight decay
                # TODO: Test SGD / batched gradient update

    def feature_vec(self, _class, sample):
        return [ff(sample[i], _class) for i, ff in enumerate(self.features_functions)]

    def get_data(self):
        """Opens and saves files according to given file path."""
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