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


class NaiveBayes:

    def __init__(self):

        # Set path variables
        self.file_path = sys.argv[1]
        self.param_path = sys.argv[2]
        self.classes = next(os.walk(self.file_path))[1]

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

    def create_class_defaultdict(self):
        """Helper method to create defaultdict of classes."""
        return {_class: defaultdict(int) for _class in self.classes}

    def count_frequencies(self):
        """Count word frequencies, word frequencies given the class and class frequencies."""

        # Iterate over each class' files
        for root, dirs, files in os.walk(self.file_path):
            for _class in self.classes:
                if root.endswith(_class):

                    # Iterate over each file
                    for file in files:
                        with open(os.path.join(root, file), encoding="latin-1") as f:
                            email = f.read().split()

                            # Count each email's frequencies
                            self.freq_class[_class] += 1
                            for word in email:
                                self.freq_word_given_class[_class][word] += 1
                                self.freq_word[word] += 1


    def fit(self):
        """Estimates probabilities given the frequencies. Then pply backoff smoothing."""

        self.count_frequencies()

        # Estimate probability of word given class: p(w|c) = f(w,c) / sum_w' f(w',c)
        for _class in self.classes:
            frequencies = self.freq_word_given_class[_class]
            _sum = sum(frequencies.values())
            self.prob_word_given_class[_class] = {k: v/_sum for k, v in frequencies.items()}

        # Estimate probability of word: p(w) = f(w) / sum(f(w')
        _sum = sum(self.freq_word.values())
        self.prob_word = {k: v/_sum for k, v in self.freq_word.items()}

        # Estimate probability of class: p(c) = f(c) / sum(f(c'))
        _sum = sum(self.freq_class.values())
        self.prob_class = {k: v/_sum for k, v in self.freq_class.items()}

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

    def save_params(self):
        params = {"prob_word": self.prob_word,
                  "prob_class": self.prob_class,
                  "prob_word_given_class": self.prob_word_given_class,
                  "prob_word_given_class_backoff": self.prob_word_given_class_backoff
                  }
        try:
            with open(f"{self.param_path}.pickle", 'wb') as handle:
                pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Succesfully saved parameters.")
        except:
            print("Something went wrong while saving parameters.")

if __name__ == "__main__":
    assert len(sys.argv) == 3, "Call script as $ python3 train.py train-dir paramfile"
    nb = NaiveBayes()
    nb.fit()
    nb.save_params()
