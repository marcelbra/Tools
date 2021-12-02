import os
import sys
from features import (word_tag,
                      prevtag_tag,
                      prevtag_word_tag,
                      substrings_tag,
                      word_shape_tag)


class CRFTagger:
    def __init__(self, data_file, paramfile):
        self.data_file = data_file
        self.paramfile = paramfile


    def get_data(self):
        data = []
        with open(self.data_file, encoding='utf-8') as train_file:
            sentences = train_file.read().split("\n\n")
            for sent in sentences:
                words = []
                tags = []
                for e in sent.split("\n"):
                    word_tag = e.split("\t")
                    if len(word_tag) == 2:
                        words.append(word_tag[0])
                        tags.append(word_tag[1])
                data.append((words, tags))

        return data


if __name__ == '__main__':
    crf = CRFTagger(data_file=sys.argv[1], paramfile=sys.argv[2])
    data = crf.get_data()
    print((word_shape_tag(data[:2])))

