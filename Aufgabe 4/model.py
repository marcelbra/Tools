"""
P7 Experimente, Evaluierung und Tools
Aufgabe 4 - Spam classification using Naive Bayes

Group:
Marcel Braasch
Nadja Seeberg
Sinem Kühlewind (geb. Demiraslan)
"""

from collections import defaultdict
import sys
import os
import pickle
import operator
import math

class NaiveBayes:

    def __init__(self, mode):

        # Sets file path arguments depending on mode of model usage
        self.mode = mode
        self.set_arguments()

        # Variables to save frequencies to
        self.freq_word_given_class = self.create_class_defaultdict()
        self.freq_word = defaultdict(int)
        self.freq_class = defaultdict(int)

        # Variables to save ML estimators to
        self.prob_word_given_class = self.create_class_defaultdict()
        self.prob_word = defaultdict(int)
        self.prob_class = defaultdict(int)

        # Variables needed for backoff smoothing
        self.rel_freq_word_given_class = self.create_class_defaultdict()
        self.prob_word_given_class_backoff = self.create_class_defaultdict()

    def predict(self):
        for file, target in self.get_files():
            with open(f"{os.getcwd()}/test/{target}/{file}", encoding="latin-1") as f:
                email = f.read().split()

                # Get probability score for each class
                # Operating in log space because probabilities get very small
                log_score = dict()
                for _class in self.classes:
                    log_prob_class = math.log(self.prob_class[_class])
                    log_prob_doc = 0
                    for word in email:
                        prob_given_class = self.prob_word_given_class_backoff[_class]
                        # It would be bad to assume unknown have 0 probability. That would likely
                        # misclassify documents simply because a word wasnt seen during data.
                        if word not in prob_given_class: continue  # Just skip
                        log_prob_doc += math.log(prob_given_class[word])
                    log_score[_class] = log_prob_class + log_prob_doc

                # Log the prediction
                prediction = max(log_score.items(), key=operator.itemgetter(1))[0]
                self.log(target, prediction)
        self.show_score()

    def log(self, target, prediction):
        """Increase counter[0] when classified correctly else increase counter[1] by 1."""
        self.counter[target==prediction] += 1

    def show_score(self):
        print(self.counter)

    def fit(self):
        """Estimates probabilities given the frequencies. Then apply backoff smoothing."""

        if self.mode == "test":
            params = self.load_parameters()
            self.prob_word_given_class = params["prob_word_given_class"]
            self.prob_word = params["prob_word"]
            self.prob_class = params["prob_class"]
            self.prob_word_given_class_backoff = params["prob_word_given_class_backoff"]
            self.counter = [0,0]  # Preliminary counter to count how many times classification is right

        elif self.mode == "train":
            self.count_frequencies()
            self.estimate_parameters()
            self.smooth_parameters()

    def count_frequencies(self):
        """Count word frequencies, word frequencies given the class and class frequencies."""
        for file, _class in self.get_files():
            with open(os.path.join(os.getcwd(), file), encoding="latin-1") as f:
                email = f.read().split()
                # Count emails frequencies
                self.freq_class[_class] += 1
                for word in email:
                    self.freq_word_given_class[_class][word] += 1
                    self.freq_word[word] += 1

    def estimate_parameters():
        """Estimates the model parameters given the words frequencies."""

        for _class in self.classes:
            # Estimate probability of word given class: p(w|c) = f(w,c) / sum_w' f(w',c)
            frequencies = self.freq_word_given_class[_class]
            _sum = sum(frequencies.values())
            self.prob_word_given_class[_class] = {k: v/_sum for k, v in frequencies.items()}

        # Estimate probability of word: p(w) = f(w) / sum(f(w')
        _sum = sum(self.freq_word.values())
        self.prob_word = {k: v/_sum for k, v in self.freq_word.items()}

        # Estimate probability of class: p(c) = f(c) / sum(f(c'))
        _sum = sum(self.freq_class.values())
        self.prob_class = {k: v/_sum for k, v in self.freq_class.items()}

    def smooth_parameters(self):
            """Smoothes the model by Kneser-Ney smoothing."""

            # Calulcate discount factor delta after Kneser/Essen/Ney
            n = lambda x: sum(list(self.freq_word_given_class[_class].values()).count(x) for _class in self.classes)
            delta = n(1) / (n(1) + 2*n(2))

            # Calulcate relativ frequencies of word given class: r(w|c) = max(0, f(w,c) - delta) / sum_w' f(w',c)
            for _class in self.classes:
                _sum = sum(self.freq_word_given_class[_class].values())  # Sum_w' f(w',c)
                self.rel_freq_word_given_class[_class] = {k: max(0, v - delta) / _sum
                                                          for k, v in self.freq_word_given_class[_class].items()}

            # Dynamically calculate backoff factor alpha
            alpha = lambda _class: 1 - sum(self.rel_freq_word_given_class[_class].values())

            # Calulcate discount probability: p(w|c) = r(w|c) + alpha(c)p(w)
            for _class in self.classes:
                _alpha, rel_word_freq = alpha(_class), self.rel_freq_word_given_class[_class]
                self.prob_word_given_class_backoff[_class]= {k: rel_word_freq[k] + _alpha * self.prob_word[k]
                                                              for k, v in self.prob_word_given_class[_class].items()}

    def set_arguments(self):
        """Sets variables depending on the mode of model usage."""

        self.paramfile = sys.argv[1]
        self.data_dir = sys.argv[2]
        # In train mode parameters are passed vice versa
        if self.mode == "train":
            self.data_dir, self.paramfile = self.paramfile, self.data_dir
        self.classes = next(os.walk(self.data_dir))[1]

    def create_class_defaultdict(self):
        """Helper method to create defaultdict of classes."""
        return {_class: defaultdict(int) for _class in self.classes}

    def get_files(self):
        """
        Generator object that returns the next file and its
        corresponding class every time next() is called.
        """
        for root, dirs, files in os.walk(self.data_dir):
            for _class in self.classes:
                if root.endswith(_class):
                    for file in files:
                        yield file, _class

    def load_parameters(self):
        with open(self.paramfile, 'rb') as handle:
            return pickle.load(handle)

    def save_parameters(self):
        params = {"prob_word": self.prob_word,
                  "prob_class": self.prob_class,
                  "prob_word_given_class": self.prob_word_given_class,
                  "prob_word_given_class_backoff": self.prob_word_given_class_backoff}
        with open(f"{self.paramfile}.pickle", 'wb') as handle:
            pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)