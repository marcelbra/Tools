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
from tqdm import tqdm


class CRFTagger:

    def __init__(self, data_file, paramfile):

        self.data_file = data_file
        self.paramfile = paramfile
        self.tagset = self.get_tagset()
        self.epochs = 3
        self.lr = 1e-5

    def fit(self):

        # TODO: data restriction raus, n-gram wieder 3,7

        print("Load data")
        data = self.get_data()
        data = data[:100]  # Delete, only for debugging


        # Observed feature values
        weights = defaultdict(float)

        for n in range(self.epochs):
            for words, tags in tqdm(data):
                for i, word in enumerate(words):
                    if not i: continue
                    tag = tags[i]
                    prev_tag = tags[i-1]
                    feature_count = self.feature_extraction(prev_tag, tag, words, i)
                    weights = defaultdict(int, dict(weights, **feature_count))

        # Expected feature values
                    alphas = self.forward(words, weights)
                    betas = self.backward(words, weights)
                    gammas = self.get_estimated_feature_values(words, weights, alphas, betas)


    def get_estimated_feature_values(self, words, weights, alphas, betas):
        """
        Calulates gamma values for the word sequence, given alphas and betas.
        """
        gammas = self.init_scores(mode="gammas", words=words)
        for i in range(1, len(words)):
            for tag, beta_score in betas[i].items():
                for previous_tag, alpha_score in alphas[i-1].items():
                    feature_count = self.feature_extraction(previous_tag, tag, words, i)
                    feature_vector = list(feature_count.values())
                    s = self.get_score(feature_count, weights)
                    score = gammas[i][tag][previous_tag] + dot(feature_vector, s)
                    # TODO: Denke, der Score muss noch zum Exponenten genommen werden
                    score = math.exp(score)
                    #TODO: Andere Gruppe hat hier die Rechnung ohne Score, allerdings machen wir es ja nach ytt(i)
                    p = alpha_score + score + beta_score - alphas[-1]["BOUNDARY"]
                    gammas[i][tag][previous_tag] = p
        return gammas

    def forward(self, words, weights):
        alphas = self.init_scores(mode="alpha", words=words)
        for i in tqdm(range(1, len(words))):
            for tag in self.tagset:
                for previous_tag, previous_score in alphas[i-1].items():
                    feature_count = self.feature_extraction(previous_tag, tag, words, i)
                    feature_vector = list(feature_count.values())
                    #TODO: Hier kommen wir von der 0 nicht weg, weil self.get_score anfangs einen reinen 0-Vektor zurückgibt
                    score = self.get_score(feature_count, weights)
                    score = previous_score + dot(feature_vector, score)
                    alphas[i][tag] = log_sum_exp(alphas[i][tag], score)
        return alphas

    def backward(self, words, weights):
        """
        We iterate backward from n to 0. The recursion looks at the right position of i.
        In slides we said beta(i-1) is dependent on beta(i). That's equivalent to saying
        beta(i) is dependent on beta(i+1) (makes the handling of indices more convenient).
        """
        betas = self.init_scores(mode="beta", words=words)
        for i in tqdm(range(len(words) - 1)[::-1]):
            for tag in self.tagset:
                for next_tag, next_score in betas[i+1].items():
                    feature_count = self.feature_extraction(tag, next_tag, words, i)
                    feature_vector = list(feature_count.values())
                    score = self.get_score(feature_count, weights)
                    score = next_score + dot(feature_vector, score)
                    betas[i][tag] = log_sum_exp(betas[i][tag], score)
        return betas

    def get_score(self, feat_count, weights):
        return [weights[feat] if feat in weights else 0 for feat in feat_count]

    def feature_extraction(self, prevtag, tag, words, i):

        word_tag = words[i], tag
        prevtag_tag = prevtag, tag
        prevtag_word_tag = prevtag, tag, words[i]
        word_shape_tag = get_word_shape(words[i]), tag
        ngrams_tag = get_substrings_tag(tag, words)

        features = [word_tag, prevtag_tag, prevtag_word_tag,
                    word_shape_tag] + ngrams_tag

        feature_count = {str(k):v for k,v in Counter(features).items()}

        return feature_count

    def init_scores(self, mode, words):
        """Initializer for alpha, beta, gamma and weight scores."""
        if mode=="gammas":
            structure = [{tag: {tag: 1 if tag=="BOUNDARY" else 0
                                for tag in self.tagset}
                          for tag in self.tagset}
                         for _ in words]
        elif mode=="alpha" or mode=="beta":
            structure = [{tag: 1 if tag=="BOUNDARY" else 0
                          for tag in self.tagset}
                         for _ in words]
        return structure

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
