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
                   get_word_shape,
                   init_scores,
                   feature_extraction,
                   get_context_features,
                   get_lexical_features)
from collections import defaultdict
import re
from copy import copy
from tqdm import tqdm

class CRFTagger:

    def __init__(self, data_file, paramfile):
        self.data_file = data_file
        self.paramfile = paramfile
        self.tagset = self.get_tagset()
        self.weights = defaultdict(float)

    def fit(self, lr=1e-5):
        for epoch in range(3):
            for words, tags in tqdm(self.get_data()):
                #alphas = self.step(words, forward=True)
                #betas = self.step(words, forward=False)
                alphas = self.forward(words)
                betas = self.backward(words)

                estimated = self.get_estimated_frequencies(words, alphas, betas)
                observed = self.get_observed_frequencies(words, tags)
                self.weight_update(estimated, observed, lr)
        self.save_weights()

    def forward(self, words):
        values = init_scores(words, True, self.tagset)
        for i in range(1, len(words)):
            for tag in values[i].keys():
                for previous_tag, previous_score in values[i-1].items():
                    values[i][tag] += math.log(previous_score + self.score(previous_tag, tag, words, i))
        return values

    def backward(self, words):
        values = init_scores(words, False, self.tagset)
        for i in range(len(words) - 1)[::-1]:
            for tag in values[i].keys():
                for next_tag, next_score in values[i+1].items():
                    values[i][tag] += math.log(next_score + self.score(tag, next_tag, words, i))#cache[tags]
        return values

    """
    def step(self, words, forward):
        values = init_scores(words, forward, self.tagset)
        _range, direction = (range(1, len(words)), -1) if forward else (range(len(words) - 1)[::-1], 1)
        for i in _range:
            #cache = {}
            for tag in values[i].keys():
                for adjacent_tag, adjacent_score in values[i+direction].items():
                    tags = (adjacent_tag, tag) if forward else (tag, adjacent_tag)
                    #if tags not in cache:
                    #    cache[tags] = math.log(adjacent_score + self.score(*tags, words, i))
                    values[i][tag] += math.log(adjacent_score + self.score(*tags, words, i))#cache[tags]
        return values
    """

    def score(self, adjacent_tag, tag, words, i):
        feature_counts = feature_extraction(adjacent_tag, tag, words, i)
        score = sum(self.weights[feature] * counts for feature, counts in feature_counts.items())
        return math.exp(score)

    def get_estimated_frequencies(self, words, alphas, betas):
        gammas = defaultdict(float)
        for i in range(1, len(words)):
            cache = {} #(argument, tag, i)
            for tag, beta_score in betas[i].items():

                # Calculate gamma for lexical features
                lexical_features = get_lexical_features(tag, words, i)
                for feature in lexical_features:
                    gammas[feature] = math.exp(alphas[i][tag] + betas[i][tag] - alphas[-1]["<s>"])

                # Calculate gamma for context features
                lexixal_score = sum(self.weights[f] for f in lexical_features)
                for previous_tag, alpha_score in alphas[i - 1].items():

                    # Caching
                    context_features_key = ("context_features", tag, i)
                    context_score_key = ("context_score", tag, i)
                    if context_features_key not in cache:
                        cache[context_features_key] = get_context_features(previous_tag, tag, words, i)
                        if context_score_key not in cache:
                            cache[context_score_key] = sum(self.weights[f] for f in cache[context_features_key])

                    p = math.exp(alpha_score + cache[context_score_key] + lexixal_score + beta_score - alphas[-1]["<s>"])
                    for feature in cache[context_features_key]:
                        gammas[feature] += p

        return gammas

    def get_observed_frequencies(self, words, tags):
        observed_frequencies = defaultdict(float)
        for i, (word, tag) in enumerate(zip(words, tags)):
            lexical = get_lexical_features(tag, words, i)
            context = get_context_features(tags[i-1], tag, words, i)
            for feature in lexical + context:
                observed_frequencies[feature] += 1
        return observed_frequencies

    def weight_update(self, estimated, observed, lr):
        for feature, value in estimated.items():
            self.weights[feature] -= value * lr
        for feature, value in observed.items():
            self.weights[feature] += value * lr

    def get_tagset(self):
        all_tags = []
        for _, tags in self.get_data():
            all_tags.extend(tags)
        return list(set(all_tags))

    def get_data(self):
        data = []
        with open(self.data_file, encoding='utf-8') as train_file:
            file = train_file.read().split("\n\n")
            for sent in file:
                if sent != "":
                    word_tag_pairs = [line.split("\t") for line in sent.split("\n")]
                    words, tags = zip(*word_tag_pairs)
                    words = [" "] + list(words) + [" "]
                    tags = ["<s>"] + list(tags) + ["</s>"]
                data.append((words,tags))
        return data
                #yield words, tags

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

