"""
P7 Tools - Aufgabe 6
Training of CRF Tagger
Gruppe:
Marcel Braasch
Nadja Seeberg
Sinem Kühlewind (geb. Demiraslan)
"""

import os
import sys
from collections import Counter
from utils import (add, div, mul,
                   sub, dot, create_vec,
                   log_sum_exp,)
from features import (word_tag,
                      prevtag_tag,
                      prevtag_word_tag,
                      substrings_tag,
                      word_shape_tag,)
from collections import defaultdict
import re


class CRFTagger:

    def __init__(self, data_file, paramfile):

        self.data_file = data_file
        self.paramfile = paramfile
        self.tagset = self.get_tagset()
        self.feature_functions = [word_tag,
                                  prevtag_tag,
                                  prevtag_word_tag,
                                  substrings_tag,
                                  word_shape_tag]

    def fit(self):

        weights = self.init_weights()
        data = self.get_data()

        for words, tags in data:

            # Expected feature values
            alpha = self.forward(words, weights)
            betas = self.backward(words, weights)
            gammas = self.get_estimated_feature_values(words, weights, alphas, betas)

            # Observed feature values


    def get_estimated_feature_values(self, words, weights, alphas, betas):
        """
        Calulates gamma values for the word sequence, given alphas and betas.
        """
        gammas = self.init_scores(words, gammas=True)
        for i in range(1, len(words)):
            for tag, beta_score in betas[i]:
                for previous_tag, alpha_score in alphas[i-1]:
                    feature_vector = self.feature_vector(previous_tag, tag, words, i)
                    score = mul(feature_vector, weights)
                    gamma = alphas[i-1][previous_tag] + score + betas[i][tag] - alphas[-1]["<s>"]
                    gammas[i][tag][previous_tag] += gamma
        return gammas

    def forward(self, words, weights):
        alphas = self.init_scores(words)
        for i in range(1, len(words)):
            for tag in self.tagset:
                for previous_tag, previous_score in alphas[i-1].items():
                    feature_vector = self.feature_vector(previous_tag, tag, words, i)
                    score = previous_score + mul(feature_vector, weights)
                    alphas[i][tag] = log_sum_exp(alphas[i][tag], score)
        return alphas

    def backward(self, words, weights):
        """
        We iterate backward from n to 0. The recursion looks at the right position of i.
        In slides we said beta(i-1) is dependent on beta(i). That's equivalent to saying
        beta(i) is dependent on beta(i+1) (makes the handling of indices more convenient).
        """
        betas = self.init_scores(words)
        for i in range(len(words) - 1)[::-1]:
            for tag in self.tagset:
                for next_tag, next_score in betas[i+1].items():
                    feature_vector = self.feature_vector(tag, next_tag, words, i)
                    score = next_score + mul(feature_vector, weights)
                    betas[i][tag] = log_sum_exp(betas[i][tag], score)
        return betas

    def feature_vector(self, previous_tag, tag, words, i):
        pass #TODO

    def init_scores(self, words, gammas=False):
        """
        Creates a list of dictionaries for every given word.
        For every word, a score for every tag will be saved.

        For alphas and betas structure looks like this:
        [{'A': 1, 'B': 0, 'C': 0},
         {'A': 1, 'B': 0, 'C': 0},
         {'A': 1, 'B': 0, 'C': 0}]

        For gammas structure looks like this:
        [{'A': {'A': 1, 'B': 0, 'C': 0},
          'B': {'A': 1, 'B': 0, 'C': 0},
          'C': {'A': 1, 'B': 0, 'C': 0}},
         {'A': {'A': 1, 'B': 0, 'C': 0},
          'B': {'A': 1, 'B': 0, 'C': 0},
          'C': {'A': 1, 'B': 0, 'C': 0}},
         {'A': {'A': 1, 'B': 0, 'C': 0},
          'B': {'A': 1, 'B': 0, 'C': 0},
          'C': {'A': 1, 'B': 0, 'C': 0}}]
        """
        if gammas:
            structure = [{tag: {tag: 1 if tag=="A" else 0
                                for tag in tags}
                          for tag in tags}
                         for _ in words]
        else:
            structure = [{tag: 1 if tag=="BOUNDARY" else 0
                          for tag in self.tagset}
                         for _ in words]
        return structure

    def init_weights(self):
        return [1 for _ in range(len(self.feature_functions))]

    def extract_features(self, i, words, tags):
        word_to_tag = word_tag(i, words, tags)
        prevtag_to_tag = prevtag_tag(i, tags)
        prevtag_to_word_to_tag = prevtag_word_tag(i, words, tags)
        ngrams_to_tag = substrings_tag(i, words, tags)
        word_shape_to_tag = word_shape_tag(i, words, tags)
        return word_to_tag, prevtag_to_tag, prevtag_to_word_to_tag, ngrams_to_tag, word_shape_to_tag

    def get_tagset(self):  # 54 tags together with BOUNDARY
        sentences = self.get_data()
        self.tagset = list(set([re.sub("[|]", '', taglist) for sentence, sent_tags in sentences
                                for taglist in sent_tags]))
        return self.tagset

    def aposteriori(self, position):
        """
        Computes aposteriori probabilities of each t in T at position i.
        1.) gamma_t(i) = (alpha_t(i) * beta_t(i)) / (alpha_<s>(n+1)) for all t in T
        2.) gamma_tt-1(i) = (alpha_t(i-1) * z(t, t-1, [w for w in range(n+1)], i) * beta_t-1(i)) / alpha_<s>(n+1)
        for all t and t-1 in T
        :return: aposteriori probabilities
        """
        pass

    def get_data(self):
        """
        Reads data and appends boundary token <s> to start and end of sentence.
        :return: tuple(list of words, list of tags)
        """
        data = []
        with open(self.data_file, encoding='utf-8') as train_file:
            file = train_file.read().split("\n\n")
            sentences = [["<s>"] + [word_tag.split("\t")[0] for word_tag in sent.split("\n")
                                    if len(word_tag.split("\t")) == 2] + ["<s>"] for sent in file if sent != ""]
            tags = [["BOUNDARY"] + [word_tag.split("\t")[1] for word_tag in sent.split("\n")
                                    if len(word_tag.split("\t")) == 2] + ["BOUNDARY"] for sent in file if sent != ""]
            for s, t in zip(sentences, tags):
                data.append((s, t))
            self.sentences = sentences
        return data

if __name__ == '__main__':
    crf = CRFTagger("Tiger/train.txt", paramfile="paramfile.pickle")
    crf.fit()
