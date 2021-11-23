from collections import defaultdict
import sys
import os
from features import (
    avg_word_length_of,
    amount_exclamation_mark_of,
    length_of
    )
import math
from vector_operations import add, sub, mul, div, dot, create_vec


class LogLinear:

    def __init__(self, mode):

        # Set parameters
        self.mode = mode
        self.paramfile = sys.argv[1]
        self.data_dir = sys.argv[2]
        if mode == "train":
            self.data_dir, self.paramfile = self.paramfile, self.data_dir
        self.classes = ["ham", "spam"] #TODO: next(os.walk(self.data_dir))[1] # irgendwas spinnt hier bei mir

        # Parameters and feature functions
        self.features_functions = [avg_word_length_of,
                                   amount_exclamation_mark_of,
                                   length_of]
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
            for epoch in epochs:
                grad = create_vec(0, self.n)
                for sample, true_class in self.data:

                    # Feature vector, left term in slides
                    feature_score = self.feature_vec(sample, true_class)

                    # Expectation times feature vector, right term in slides
                    prob_class_given_doc = 0
                    for _class in self.classes:
                        # This is vector of sum_c' p_theta (c',d) * f(c',d)
                        Z = sum([math.exp(dot(self.theta, self.feature_vec(c, sample)))
                                 for c in self.classes])
                        class_prob = math.exp(dot(self.theta, self.feature_vec(_class, sample)))
                        prob_class_given_doc += class_prob / Z

                    # Create vector of expectations times (element wise) feature vector
                    expectation = create_vec(prob_class_given_doc, self.n)
                    expectation_times_feature = mul(expectation, self.feature_vec(_class,sample))

                    # Combine left and right term
                    score = sub(feature_score, expectation_times_feature)

                    # Update gradient
                    grad = add(grad, score)

                # Update parameters
                self.theta = add(self.theta, mul(create_vec(self.eta, n), grad))

                # TODO: Evaluation / logging after every gradient update
                # TODO: feature functions
                # TODO: weight decay / hyperparameter tuning
                # TODO: test SGD / batched gradient update

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