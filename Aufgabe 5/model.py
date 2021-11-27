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
        self.feature_functions = [avg_word_length_pos,
                                  avg_word_length_neg,
                                  length_neg,
                                  length_pos,
                                  amount_exclamation_mark_pos,
                                  amount_exclamation_mark_neg]
        self.n = len(self.feature_functions)
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

            epochs = range(2)
            grad = create_vec(0, self.n)
            for epoch in epochs:

                scores = create_vec(0, self.n)
                for sample, true_class in self.data:

                    # Feature vector, left term in slides
                    feature_score_left = self.feature_vec(sample, true_class)
                    # Normalization constant for expectation
                    Z = sum([math.exp(dot(self.theta, self.feature_vec(c, sample))) for c in self.classes])

                    # Expectation times feature vector which will be build up by procedure below
                    expectation_times_feature_vec = create_vec(0, self.n)

                    for _class in self.classes:

                        # This is the numerator in the expectation, constant across vector entries
                        class_prob = math.exp(dot(self.theta, self.feature_vec(_class, sample)))

                        # Build up the vector entries of the "expect * feat vec"-term.
                        # This is the part right of the expectation. Each entry requires
                        # its own feature function.
                        for i in range(self.n):

                            # Feature score right of the expectation
                            feature_score_right = self.feature_functions[i](sample, _class)

                            # Combine all terms
                            expectation_times_feature_vec[i] += (class_prob * feature_score_right) / Z

                    # Combine left and right term
                    score = sub(feature_score_left, expectation_times_feature_vec)
                    # Accumulate gradient
                    scores = add(scores, score)

                # Update gradient
                grad = add(grad, scores)
                # Update parameters
                self.theta = add(self.theta, mul(create_vec(self.eta, self.n), grad))

                # TODO: Evaluation / logging after every gradient update
                # TODO: Weight decay
                # TODO: Test SGD / batched gradient update

    def feature_vec(self, sample, _class):
        return [ff(sample, _class) for ff in self.feature_functions]

    def get_data(self):
        """Opens and saves files according to given file path."""
        data = []
        for root, dirs, files in os.walk(self.data_dir):
            for _class in self.classes:
                if root.endswith(_class):
                    for file in files:
                        with open(f"train/{_class}/{file}", encoding="latin-1") as f:
                            data.append((f.read().split(), _class))
        return data

    def create_class_defaultdict(self, data_type):
        """Helper method to create defaultdict of classes."""
        return {_class: defaultdict(data_type) for _class in self.classes}