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
import math
from collections import Counter
from utils import (add, div, mul, log_sum_exp,
                   sub, dot, create_vec,
                   get_substrings_tag,
                   get_word_shape,)
from collections import defaultdict
import re

class CRFTagger:

    def __init__(self, data_file, paramfile):
        self.data_file = data_file
        self.paramfile = paramfile
        self.tagset = self.get_tagset()
        self.weights = defaultdict(float)

    def fit(self, learning_rate=1e-5):
        for epoch in range(3):
            for words, tags in self.get_data():
                for i, word in enumerate(words):
                    if not i: continue
                    alphas = self.forward(words)
                    betas = self.backward(words)
                    estimated_frequencies = self.get_estimated_feature_values(words, alphas, betas)
                    observed_frequencies = self.feature_extraction(tags[i-1], tags[i], words, i)
                    self.weight_update(estimated_frequencies, observed_frequencies, learning_rate)
        self.save_weights()

    def weight_update(self, estimated_frequencies, observed_frequencies, learning_rate):
        for feature, value in estimated_frequencies.items():
            self.weights[feature] -= value * learning_rate
        for feature, value in observed_frequencies.items():
            self.weights[feature] += value * learning_rate

    def get_estimated_feature_values(self, words, alphas, betas):
        """
        Calulates gamma values for the word sequence, given alphas and betas.
        """
        gammas = defaultdict(float)
        for i in range(1, len(words)):
            for tag, beta_score in betas[i].items():

                # Calculate gamma for lexical features
                lexical_features = self.get_lexical_features(tag, words, i)
                for feature in lexical_features:
                    gammas[feature] = math.exp(alphas[i][tag] + betas[i][tag] - alphas[-1]["BOUNDARY"])

                # Calculate gamma for context features
                for previous_tag, alpha_score in alphas[i-1].items():
                    frequencies = self.feature_extraction(previous_tag, tag, words, i)
                    context_features = self.get_context_features(previous_tag, tag, words, i)
                    score = math.exp(sum(frequencies[f"{feature}"] * self.weights[feature] for feature in context_features))
                    for feature in context_features:
                        gammas[feature] = math.exp(alphas[i - 1][tag] + betas[i][previous_tag]
                                                   + score - alphas[-1]["BOUNDARY"])
        return gammas

    def forward(self, words):
        alphas = self.init_scores(words)
        for i in range(1, len(words)):
            for tag in self.tagset:
                for previous_tag, previous_score in alphas[i-1].items():
                    alphas = self.step(alphas, previous_tag, tag, words, i, previous_score)
        return alphas

    def backward(self, words):
        betas = self.init_scores(words)
        for i in range(len(words) - 1)[::-1]:
            for tag in self.tagset:
                for next_tag, next_score in betas[i+1].items():
                    betas = self.step(betas, next_tag, tag, words, i, next_score)
        return betas

    def step(self, values, previous_next_tag, tag, words, i, prev_next_score):
        feature_count = self.feature_extraction(previous_next_tag, tag, words, i)
        feature_vector = list(feature_count.values())
        score = self.get_score(feature_count)
        score = prev_next_score + dot(feature_vector, score)
        values[i][tag] = log_sum_exp(values[i][tag], score)
        return values

    def get_score(self, feat_count):
        return [self.weights[feat] if feat in self.weights else 0 for feat in feat_count]

    def get_lexical_features(self, tag, words, i):
        word_tag = words[i], tag
        word_shape_tag = get_word_shape(words[i]), tag
        ngrams_tag = get_substrings_tag(tag, words)
        return [word_tag, word_shape_tag] + ngrams_tag

    def get_context_features(self, prevtag, tag, words, i):
        prevtag_tag = prevtag, tag
        prevtag_word_tag = prevtag, tag, words[i]
        return [prevtag_tag, prevtag_word_tag]

    def feature_extraction(self, prevtag, tag, words, i):
        lexical = self.get_lexical_features(tag, words, i)
        context = self.get_context_features(prevtag, tag, words, i)
        features = lexical + context
        feature_count = {str(k):v for k,v in Counter(features).items()}
        return feature_count

    def init_scores(self, words):
        """Initializer for alpha, beta, gamma and weight scores."""
        return [{tag: 1 if tag == "BOUNDARY" else 0 for tag in self.tagset} for _ in words]

    def get_tagset(self):  # 54 tags together with BOUNDARY
        sentences = self.get_data()
        self.tagset = list(set([re.sub("[|]", '', taglist) for sentence, sent_tags in sentences
                                for taglist in sent_tags]))
        return self.tagset

    def get_data(self):
        """
        Reads data and appends boundary token <s> to start and end of sentence.
        :return: tuple(list of words, list of tags)
        """
        with open(self.data_file, encoding='utf-8') as train_file:
            file = train_file.read().split("\n\n")
            for sent in file:
                if sent != "":
                    words, tags = ["<s>"], ["BOUNDARY"]
                    for word_tag in sent.split("\n"):
                        if len(word_tag.split("\t")) == 2:
                            words.append(word_tag.split("\t")[0])
                            tags.append(word_tag.split("\t")[1])
                    words.append("<s>")
                    tags.append("BOUNDARY")
                yield words, tags

    def save_weight(self):
        data = {"parameters": self.weights,
                "tagset": self.tagset}
        with open(self.paramfile, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    train_file = sys.argv[1]
    param_file = sys.argv[2]
    crf = CRFTagger(train_file, param_file)
    crf.fit()