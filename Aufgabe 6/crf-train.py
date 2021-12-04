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

from features import (word_tag,
                      prevtag_tag,
                      prevtag_word_tag,
                      substrings_tag,
                      word_shape_tag)
from collections import defaultdict
import re


class CRFTagger:
    def __init__(self, data_file, paramfile):
        self.data_file = data_file
        self.paramfile = paramfile
        self.forward_table = [defaultdict(int)]
        self.tagset = None
        self.sentences = []
        self.theta = defaultdict()
        # We need to store tagset in self as we need to iterate over each tag in tagset
        # in each function. Same goes for self.sentences.

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

    def fit(self):
        attributes = []
        data = self.get_data()
        for (words, tags) in data:
            feature_count = {}
            for i in range(0, len(tags)): #ich bekomme einen Fehler, wenn ich mit len(tags) +1 iteriere
                prevtag = tags[i-1]
                # extract features and save them to list as Strings
                features = self.extract_features(i, words, tags)
                for feat in features:
                    attributes.append(str(feat))
                #extract feature frequency per sentence key = feature, value = freq (as discussed in class)
                feature_count = Counter(attributes)

            feat_vec = feature_count.values()





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

    def forward(self):
        """
        Computes forward probabilities of each t in T at position i.
        1.) Initialize alpha_t(0) with 1 if t==<s> else 0
        2.) alpha_t(i) = sum([((alpha_t-1(i-1)) * z(t-1, t, [w for w in range(n+1)], i)] for t in T)
        whereas z(t-1, t, [w for w in range(n+1)], i)
        (n+1 because in Python iteration starts at 0 instead 1 as in lectures)
        = sum of all scores ([theta * fv for fv in feature_vector]) of one word  to the exponent.
        :return: forward probabilities
        """
        all_tags = self.get_tagset()
        """for sentence in self.sentences:
            for i in range(len(sentence)):
                for (tagpos, tag) in enumerate(all_tags):
                    prevtag = all_tags[tagpos - 1]
                    alpha_prevtag_prevpos = self.forward(position - 1)
                    ff1 = word_tag(prevtag) ? what
        """
        return

    def backward(self, position):
        """
        Computes backward probabilities of each t in T at position i.
        1.) Initialize beta_t(n+1) = 1 if t==<s> else 0 (n+1 is EOS)
        2.) beta_t(i-1) = sum([((beta_t-1(i)) * z(t, t-1, [w for w in range(n+1)], i)] for t in T)
        z is same as in def forward().
        :return: backward probabilities
        """
        pass

    def aposteriori(self, position):
        """
        Computes aposteriori probabilities of each t in T at position i.
        1.) gamma_t(i) = (alpha_t(i) * beta_t(i)) / (alpha_<s>(n+1)) for all t in T
        2.) gamma_tt-1(i) = (alpha_t(i-1) * z(t, t-1, [w for w in range(n+1)], i) * beta_t-1(i)) / alpha_<s>(n+1)
        for all t and t-1 in T
        :return: aposteriori probabilities
        """
        pass


if __name__ == '__main__':
    crf = CRFTagger("Tiger/train.txt", paramfile="paramfile.pickle")
    crf.forward()
    crf.fit()
