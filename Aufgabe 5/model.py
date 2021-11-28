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
import pickle
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
            self.data_dir, self.paramfile = self.data_dir, self.paramfile
        elif mode == "test":
            self.paramfile, self.data_dir = self.paramfile, self.data_dir
        self.classes = next(os.walk(self.data_dir))[1]

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
        self.my = 1e-3
        self.delta = 0

    def predict(self):
        if self.mode == "train":
            pass

        elif self.mode == "test":
            predictions = []
            data = self.get_data(self.data_dir)

        for sample, _class in data:
            feature_score = self.feature_vec(sample, _class)
            prediction = dot(feature_score, self.theta)
            if prediction >= 0:
                predictions.append("Ham")
            else:
                predictions.append("Spam")

        with open("prediction_list.txt", "w") as output_file:
            output_file.write(" ".join([pred + "\n" for pred in predictions]))

    def fit(self):
        """Estimates probabilities given the frequencies. Then apply backoff smoothing."""

        if self.mode == "test":
            pass

        elif self.mode == "train":
            last_update = {}
            timestamp_cnt = 0

            data = self.get_data(self.data_dir)

            epochs = range(2)
            grad = create_vec(0, self.n)
            for epoch in epochs:
                timestamp_cnt += 1

                scores = create_vec(0, self.n)
                for sample, true_class in data:

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
                # calculate weight decay
                self.delta = [weight * self.my for weight in self.theta]
                self.theta = add(self.theta, mul(create_vec(self.eta, self.n), sub(grad, self.delta)))

                # TODO: Evaluation / logging after every gradient update
                # TODO: Weight decay
                # TODO: Test SGD / batched gradient update

    def feature_vec(self, sample, _class):
        return [ff(sample, _class) for ff in self.feature_functions]

    def get_data(self, dir):
        """Opens and saves files according to given file path."""
        data = []
        for root, dirs, files in os.walk(self.data_dir):
            for _class in self.classes:
                if root.endswith(_class):
                    for file in files:
                        with open(f"{dir}/{_class}/{file}", encoding="latin-1") as f:
                            data.append((f.read().split(), _class))
        return data

    def create_class_defaultdict(self, data_type):
        """Helper method to create defaultdict of classes."""
        return {_class: defaultdict(data_type) for _class in self.classes}

    def save_parameters(self):
        with open(self.paramfile, "wb") as save_file:
            pickle.dump(self.theta, save_file)

    def load_parameters(self):
        with open(self.paramfile, "rb") as load_file:
            self.theta = pickle.load(load_file)
