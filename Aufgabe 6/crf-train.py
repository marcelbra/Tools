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

        data = self.get_data()

        for words, tags in data:
            weights = self.init_scores(mode="weight")

            # Observed feature values
            for ix, word in enumerate(words):
                tag = tags[ix]
                prev_tag = tags[ix-1]
                weights = {feat: 0 for feat in self.feature_extraction(prev_tag, tag, words, ix).keys()}
                feature_count = self.feature_extraction(prev_tag, tag, words, ix)
                feat_vec = self.feature_vector(feature_count)

                # Expected feature values
            alphas = self.forward(words, weights)
            betas = self.backward(words, weights)
            gammas = self.get_estimated_feature_values(words, weights, alphas, betas)


    def get_estimated_feature_values(self, words, weights, alphas, betas):
        """
        Calulates gamma values for the word sequence, given alphas and betas.
        """
        gammas = self.init_scores(words, mode="gammas")
        for i in range(1, len(words)):
            for tag, beta_score in betas[i]:
                for previous_tag, alpha_score in alphas[i-1]:
                    feature_count = self.feature_extraction(previous_tag, tag, words, i)
                    feature_vector = self.feature_vector(feature_count)
                    weights_for_score = self.get_weights_for_scor(feature_count, weights)
                    score = mul(feature_vector, weights_for_score)
                    #gamma = alphas[i-1][previous_tag] + score + betas[i][tag] - alphas[-1]["<s>"]
                    p = alpha_score + score + beta_score - alphas[-1]["<s>"]
                    #gammas[i][tag][previous_tag] += gamma
                    gammas += p * feature_vector
        return gammas

    def forward(self, words, weights):
        alphas = self.init_scores(words, mode="alpha")
        for i in range(1, len(words)):
            for tag in self.tagset:
                for previous_tag, previous_score in alphas[i-1].items():
                    feature_count = self.feature_extraction(prevtag_tag, tag, words, i)
                    feature_vector = self.feature_vector(feature_count)
                    weights_for_score = self.get_weights_for_scor(feature_count, weights)
                    score = mul(feature_vector, weights_for_score)
                    score = previous_score + mul(feature_vector, weights_for_score)
                    alphas[i][tag] = log_sum_exp(alphas[i][tag], score)
        return alphas

    def backward(self, words, weights):
        """
        We iterate backward from n to 0. The recursion looks at the right position of i.
        In slides we said beta(i-1) is dependent on beta(i). That's equivalent to saying
        beta(i) is dependent on beta(i+1) (makes the handling of indices more convenient).
        """
        betas = self.init_scores(words, mode="beta")
        for i in range(len(words) - 1)[::-1]:
            for tag in self.tagset:
                for next_tag, next_score in betas[i+1].items():
                    feature_count = self.feature_extraction(tag, next_tag, words, i)
                    feature_vector = self.feature_vector(feature_count)
                    weights_for_score = self.get_weights_for_scor(feature_count, weights_for_score)
                    score = mul(feature_vector, weights_for_score)
                    score = next_score + mul(feature_vector, weights)
                    betas[i][tag] = log_sum_exp(betas[i][tag], score)
        return betas

    def get_weights_for_scor(self, feat_count, weights):
        weights_for_score = []
        for feature in feat_count:
            for feat in weights:
                if feature == feat:
                    weights_for_score.append(weights[feat])
        return weights_for_score


    def feature_extraction(self, prevtag, tag, words, i):
        features = []
        word_to_tag = word_tag(tag, words, i)
        features.append(str(word_to_tag))
        prevtag_to_tag = prevtag_tag(prevtag, tag, i)
        features.append(str(prevtag_to_tag))
        prevtag_to_word_to_tag = prevtag_word_tag(prevtag, tag, words, i)
        features.append(str(prevtag_to_word_to_tag))
        ngrams_tag = substrings_tag(tag, words)
        features.extend(ngrams_tag)
        word_shape_to_tag = word_shape_tag(tag, words, i)
        features.append(str(word_shape_to_tag))

        feature_count = Counter(features)
        return feature_count

    def feature_vector(self, prevtag_tag, tag, words, i):
        extracted_freqs = self.feature_extraction(prevtag_tag, tag, words, i)
        feature_vec = list(extracted_freqs.values())
        return feature_vec

    def init_scores(self, mode, *tags, **words):
        """Initializer for alpha, beta, gamma and weight scores."""
        structure = None
        if mode=="gamma":
            structure = [{tag: {tag: 1 if tag=="A" else 0
                                for tag in tags}
                          for tag in tags}
                         for _ in words]
        elif mode=="alpha" or mode=="beta":
            structure = [{tag: 1 if tag=="BOUNDARY" else 0
                          for tag in self.tagset}
                         for _ in words]
        elif mode=="weight":
            structure = defaultdict()
        return structure

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
