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
                   sub, dot, sign, create_vec,
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
                # alphas = self.step(words, forward=True)
                # betas = self.step(words, forward=False)

                # precompute values
                lexical_scores = self.get_lexical_scores(words)
                tags = self.get_tags(words, lexical_scores)
                scores = self.get_context_scores(scores, tags)
                scores.update(lexical_scores)

                alphas = self.forward(words, scores, tags)
                betas = self.backward(words, scores, tags)
                estimated = self.get_estimated_frequencies(words, alphas, betas)
                observed = self.get_observed_frequencies(words, tags)
                self.weight_update(estimated, observed, lr, mu)

                #self.viterbi(words, scores)

            self.save_weights()

    def viterbi(self, words, scores):
        tags = []
        viterbi_scores = [{} for _ in range(len(words))]
        best_prev_tag = deepcopy(viterbi_scores)
        viterbi_scores[0] = {"<s>": 0}
        for i in range(1, len(words)):
            for tag in self.tagset:
                for previous_tag, previous_score in viterbi_scores[i - 1].items():
                    score = math.log(previous_score + scores[(previous_tag, tag, i)])



    def forward(self, words, scores, tags=None):
        values = init_scores(words, True, self.tagset)
        threshold = math.log(0.001)
        max_lex_score = 0
        for i in range(1, len(words)):
            for tag in values[i].keys():
                """
                # First compute lexical scores to find out maximum lexical score.
                # Maximum lexical score serves as threshold to iterate over smaller number of tags.
                lexical_features = Counter(get_lexical_features(tag, words, i))
                current_lex_score = sum(self.weights[feature] * counts for feature, counts in lexical_features.items())
                current_lex_score = max_lex_score if current_lex_score > max_lex_score else current_lex_score
                """
                # TODO lexikalischen Scores im Cache speichern
                for previous_tag, previous_score in values[i - 1].items():
                    """if current_lex_score > (max_lex_score + threshold):
                        # TODO hier bei der Berechnung des scores, lex Score nicht erneut ausrechnen \
                        # TODO sondern aus dem Cache holen -> Funktion score() bearbeiten -> greift auf feature_extraction() zurück
                        # TODO -> EVTL: Fallunterscheidung in feature_extraction, um nur die lexikalischen Features zu extrahieren
                    """
                    values[i][tag] += math.log(previous_score + scores[(tag, previous_tag,i)])
        return values

    def backward(self, words, scores, tags=None):
        values = init_scores(words, False, self.tagset)
        for i in range(len(words)-1, 0, -1):
            for tag in values[i].keys():
                for next_tag, next_score in values[i].items():
                    values[i][tag] += math.log(next_score + scores[(next_tag, tag,i)])#
        return values

    def get_lexical_scores(self, words):
        """Given the words computes all lexical scores and stores
        them in dict as: {(tag, i): lexical_score(tag,i), ... }"""
        pass

    def get_context_scores(self, words, tags):
        cache = {}
        for i in range(1, len(words)):
            for tag in self.tagset:
                for other_tag in self.tagset:
                    key = (tag, other_tag, i)
                    if key not in cache:
                        self.scores_cache[key] = self.score(other_tag, tag, words, i)
        return cache

    def get_tags(self, words, lexical_scores):
        """Given the lexical scores find out which tag sequence is the best."""
        pass
        """
        tags = [['<s>']]
        for i in range(1, len(words)-1):
            tags = copy(self.tagset)
            # TODO: Hier statt die besten 10, die auswählen, die die Bedingungen erfüllen
            tags.sort(key=lambda t: score[(words[i], t)], reverse=True)
            tags.append(tags[:10])
        tags.append('</s>')
        return tags
        """

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
            cache = {}  # (argument, tag, i)
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

                    p = math.exp(
                        alpha_score + cache[context_score_key] + lexixal_score + beta_score - alphas[-1]["<s>"])
                    for feature in cache[context_features_key]:
                        gammas[feature] += p

        return gammas

    def get_observed_frequencies(self, words, tags):
        observed_frequencies = defaultdict(float)
        for i, (word, tag) in enumerate(zip(words, tags)):
            lexical = get_lexical_features(tag, words, i)
            context = get_context_features(tags[i - 1], tag, words, i)
            for feature in lexical + context:
                observed_frequencies[feature] += 1
        return observed_frequencies

    """
    def weight_update(self, estimated, observed, lr):
        for feature, value in estimated.items():
            self.weights[feature] -= value * lr
        for feature, value in observed.items():
            self.weights[feature] += value * lr  
    """

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


if __name__ == '__main__':
    train_file = r"Tiger/train.txt"#sys.argv[1]       # TODO Argumente durch sys.argvs ersetzen
    param_file = "paramfile.pickle"#.argv[2]
    crf = CRFTagger(train_file, param_file)
    crf.fit()
