"""
P7 Tools - Aufgabe 6
Training of CRF Tagger

Gruppe:
Marcel Braasch
Nadja Seeberg
Sinem Kühlewind (geb. Demiraslan)
"""

import os
import pickle
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
from copy import copy, deepcopy
from tqdm import tqdm


class CRFTagger:

    def __init__(self, data_file, paramfile):
        self.data_file = data_file
        self.paramfile = paramfile
        self.tagset = self.get_tagset()
        self.weights = defaultdict(float)

    def fit(self,lr=1e-5, mu=1e-5):
        data = self.get_data()
        for epoch in range(2):
            for words, tags in tqdm(data):

                # precompute values
                lexical_features, lexical_scores = self.get_lexical(words)
                tags = self.get_tags(words, lexical_scores)
                context_features, scores = self.get_context(words, tags)
                scores.update(lexical_scores)
                features = {"lexical": lexical_features,
                            "context": context_features}

                alphas = self.forward(words, scores, tags)
                betas = self.backward(words, scores, tags)
                estimated = self.get_estimated_frequencies(words, alphas, betas, scores, features)
                observed = self.get_observed_frequencies(words, tags, scores, features)
                self.weight_update(estimated, observed, lr, mu)
                #self.viterbi(words, scores)

            self.save_weights()

    def forward(self, words, scores, tags=None):
        values = init_scores(words, True, self.tagset)
        for i in range(1, len(words)):
            for tag in values[i].keys():
                for previous_tag, previous_score in values[i - 1].items():
                    values[i][tag] += math.log(previous_score + scores[(tag, previous_tag,i)])
        return values

    def backward(self, words, scores, tags=None):
        values = init_scores(words, False, self.tagset)
        for i in range(len(words)-1, 0, -1):
            for tag in values[i].keys():
                for next_tag, next_score in values[i].items():
                    values[i][tag] += math.log(next_score + scores[(next_tag, tag,i)])#
        return values

    def viterbi(self, words, scores):
        viterbi_scores = [{} for _ in range(len(words))]
        best_prev_tag = deepcopy(viterbi_scores)
        viterbi_scores[0] = {"<s>": 0}
        for i in range(1, len(words)):
            for tag in self.tagset:
                for previous_tag, previous_score in viterbi_scores[i - 1].items():
                    score = math.log(previous_score + scores[(previous_tag, tag, i)])
                    if tag not in viterbi_scores[i]:
                        viterbi_scores[i][tag] = score
                        best_prev_tag[i][tag] = previous_tag
                    elif tag in viterbi_scores[i] and score > viterbi_scores[i][tag]:
                        viterbi_scores[i][tag] = score
                        best_prev_tag[i][tag] = previous_tag

        tags = ["</s>"]
        tags += [best_prev_tag[i][tag] for i in range(len(words)-1, 0, -1)]

        return tags[::-1]

    def score(self, adjacent_tag, tag, words, i, mode):
        features = feature_extraction(adjacent_tag, tag, words, i, mode)
        score = math.exp(sum(self.weights[feature] * counts for feature, counts in features.items()))
        return features, score

    def get_estimated_frequencies(self, words, alphas, betas, scores, features):
        gammas = defaultdict(float)
        for i in range(1, len(words)):
            for tag, beta_score in betas[i].items():

                # Calculate gamma for lexical features
                lexical_key = (tag,i)
                lexical_features = features["lexical"][lexical_key]
                lexixal_score = scores[lexical_key]
                for feature in lexical_features.keys():
                    gammas[feature] = math.exp(alphas[i][tag] + betas[i][tag] - alphas[-1]["<s>"]) # TODO: Müsste ["<s>"] = ["</s>"] sein?

                # Calculate gamma for context features
                for previous_tag, alpha_score in alphas[i-1].items():
                    context_key = (previous_tag,tag,i)
                    context_features = features["context"][context_key]
                    context_score = scores[context_key]
                    p = math.exp(alpha_score + context_score + lexixal_score + beta_score - alphas[-1]["<s>"])
                    for feature in context_features:
                        gammas[feature] += p

        return gammas

    def get_observed_frequencies(self, words, tags, scores, features):
        observed_frequencies = defaultdict(float)
        for i, (word, tag) in enumerate(zip(words, tags)):
            lexical_key = (tag, i)
            context_key = (tags[i-1], tag, i)
            all_features = list(features["lexical"][lexical_key].keys())
            if i > 0:
                all_features += list(features["context"][context_key].keys())
            for feature in all_features:
                observed_frequencies[feature] += 1
        return observed_frequencies

    def get_lexical(self, words):
        feature_cache, score_cache = {}, {}
        for i in range(len(words)):
            for tag in self.tagset:
                key = (tag, i)
                if key not in score_cache:
                    features, score = self.score(None, tag, words, i, mode="lexical")
                    feature_cache[key] = features
                    score_cache[key] = score
        return feature_cache, score_cache

    def get_context(self, words, tags):
        feature_cache, score_cache = {}, {}
        for i in range(1, len(words)):
            for tag in tags:
                for other_tag in tags:
                    key = (tag, other_tag, i)
                    if key not in score_cache:
                        features, score = self.score(other_tag, tag, words, i, mode="context")
                        feature_cache[key] = features
                        score_cache[key] = score
        return feature_cache, score_cache

    def get_tags(self, words, lexical_scores):
        """Given the lexical scores find out which tag sequence is the best."""
        threshold = math.log(0.001)
        max_lex_score = max(lexical_scores.values())
        tags = self.tagset.copy()
        #for i in range(1, len(words)-1):
        #    for tag in tags:
        #        if lexical_scores[(tag, i)] + threshold < max_lex_score:
        #            tags.remove(tag)
        return tags#[["<s>"]] + tags + [["</s>"]]

    def weight_update(self, estimated, observed, lr, mu):
        for feature, value in (list(estimated.items()) + list(observed.items())):
            weight = self.weights[feature]
            weight_sign = math.copysign(1, weight)
            #TODO: Müsste hier nicht weight_sign statt weight genommen werden?
            delta = math.copysign(mu, weight)
            #TODO: Wo ist die Unterscheidung ob wir delta subtrahieren oder addieren je nach Vorzeichen des Gewichts?
            self.weights[feature] -= (value + delta) * lr
            new_weight_sign = math.copysign(1, self.weights[feature])
            if weight_sign != new_weight_sign:
                self.weights[feature] = 0

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
                data.append((words, tags))
        return data
        # yield words, tags

    def save_weights(self):
        data = {"parameters": self.weights,
                "tagset": self.tagset}
        with open(self.paramfile, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

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


if __name__ == '__main__':
    train_file = r"Tiger/train.txt"#sys.argv[1]       # TODO Argumente durch sys.argvs ersetzen
    param_file = "paramfile.pickle"#.argv[2]
    crf = CRFTagger(train_file, param_file)
    crf.fit()
