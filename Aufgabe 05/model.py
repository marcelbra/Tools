"""
P7 Experimente, Evaluierung und Tools
Aufgabe 5 - Spam classification using log linear models
Group:
Marcel Braasch
Nadja Seeberg
Sinem Kühlewind (geb. Demiraslan)
"""

from collections import defaultdict, Counter
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
        _, self.classes, _ = next(os.walk(self.data_dir))

        # Parameters and feature functions
        self.feature_functions = [avg_word_length_pos,
                                  avg_word_length_neg,
                                  length_neg,
                                  length_pos,
                                  amount_exclamation_mark_pos,
                                  amount_exclamation_mark_neg]
        self.n = len(self.feature_functions)

    def predict(self, mode, theta, write_data=None, data_dir=None):
        """
        Predicts spam or ham given the set to evaluate on and the parameters.
        If wished user can pass the directory of the data. If not given the mode
        the methods infers whether to pick the dev (for hyperparameter tuning) or
        the test set (for evaluation).
        """

        if not data_dir:
            # Gives the user the opportunity to pass an own dataset
            # If it is not the default dev or test set
            data_dir = "dev" if mode=="dev" and not data_dir else "test"

        data = self.get_data(data_dir)
        counter = [0,0]
        predictions = "" if write_data else None

        for sample, true_class in data:

            # Saves which class is the best
            max_score, max_class = float("-inf"), None

            # Wir haben Ihren Vorschlag "Die predict-Funktion sollte auch auf Dateien angewendet werden können,
            # die noch nicht klassifiziert sind" gesehen, aber sind uns unsicher, was wir verändern sollen.
            # Wenn wir das Model auf Klassen x,y trainieren, dann wird die Inferenz doch auch auf genau diesen
            # Klassen stattfinden, oder?
            for _class in self.classes:

                prediction_for_class = dot(self.feature_vec(sample, _class), theta)
                if prediction_for_class > max_score:
                    max_score = prediction_for_class
                    max_class = _class

            # Save predictions for writing in the end
            if predictions is not None:
                predictions += f"True class: {true_class}. Predicted class: {max_class}.\n"

            counter[true_class==max_class] += 1

        if predictions is not None:
            with open("prediction.txt", "w", encoding="utf-8") as f:
                f.write(predictions)

        accuracy = counter[True] / (counter[False] + counter[True])

        return accuracy

    def fit(self,
            eta=1e-5,
            epochs=30,
            ):
        """
        Fits the LL model using gradient ascent on log likelihood of the data.
        Performs ordinary gradient ascent to do the weight update and punishes
        large weights by l2 regularized likelihood.
        For a reference on what we calulcate here see "gradient.jpeg".
        """

        data = self.get_data(self.data_dir)
        theta = [0] * self.n
        grad = create_vec(0, self.n)
        best_theta, best_score = theta, float("-inf")
        print(self.predict(mode="test", theta=theta), "(Starting score)")

        for epoch in range(epochs):

            grad_step = create_vec(0, self.n)
            for sample, true_class in data:

                # Feature vector, left term in slides
                feature_score_left = self.feature_vec(sample, true_class)

                # Normalization constant for expectation
                Z = sum([math.exp(dot(theta, self.feature_vec(sample, c))) for c in self.classes])

                # Expectation times feature vector which will be build up by procedure below, right term in slides
                expectation_times_feature_vec = create_vec(0, self.n)

                for _class in self.classes:

                    # This is the numerator in the expectation, constant across vector entries
                    class_prob = math.exp(dot(theta, self.feature_vec(sample, _class)))

                    for i in range(self.n):
                        # Feature score right of the expectation
                        feature_score_right = self.feature_functions[i](sample, _class)
                        # Combine all terms
                        expectation_times_feature_vec[i] += (class_prob * feature_score_right) / Z

                # Combine left and right term
                grad_step_i = sub(feature_score_left, expectation_times_feature_vec)
                # Accumulate gradient
                grad_step = add(grad_step, grad_step_i)

            # Update gradient
            grad = add(grad, grad_step)

            # Do finetuning on the held out dev set to determine mu
            mu = self.determine_mu(grad=grad, theta=theta, eta=eta)

            # Calculate weight decay parameter
            delta = mul(create_vec(mu, self.n), theta)

            # Update parameters
            theta = add(theta, mul(create_vec(eta, self.n), sub(grad, delta)))

            # Evaluate on test set
            eval_score = self.predict(mode="test", theta=theta)
            print(f"Score for epoch {epoch}: {eval_score}")

            # Saves the best setting
            if eval_score > best_score:
                best_score = eval_score
                best_theta = theta

        # Save the best test data predictions
        self.predict(mode="test", theta=best_theta, write_data=True)

        # Return trained parameters when done
        return best_theta

    def determine_mu(self, grad, theta, eta):
        """
        Performs exponential hyperparameter search for mu given
        the current model and its learning rate.
        """
        steps = list(map(lambda x: x/2, list(range(0,5))))
        best_mu, best_acc = None, float("-inf")
        for step in steps:
            mu = math.exp(-step)
            delta = mul(create_vec(mu, self.n), theta)
            theta = add(theta, mul(create_vec(eta, self.n), sub(grad, delta)))
            acc = self.predict(mode="dev", theta=theta)
            if best_acc < acc:
                best_acc = acc
                best_mu = mu
        return best_mu

    def feature_vec(self, sample, _class):
        return [ff(sample, _class) for ff in self.feature_functions]

    def get_data(self, dir):
        """Opens and saves files according to given file path."""
        data = []
        for root, dirs, files in os.walk(dir):
            for _class in self.classes:
                if root.endswith(_class):
                    for file in files:
                        with open(f"{dir}/{_class}/{file}", encoding="latin-1") as f:
                            data.append((f.read().split(), _class))
        return data

    def create_class_defaultdict(self, data_type):
        """Helper method to create defaultdict of classes."""
        return {_class: defaultdict(data_type) for _class in self.classes}

    def save_parameters(self, theta):
        with open(self.paramfile, "wb") as save_file:
            pickle.dump(theta, save_file)

    def load_parameters(self):
        with open(self.paramfile, "rb") as load_file:
            theta = pickle.load(load_file)
        return theta

