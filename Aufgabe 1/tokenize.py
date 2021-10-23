#!/usr/bin/env python

import re
import string


class HTMLTokenizer:

    def __init__(self, abbreviations, in_file):
        with open(abbreviations, 'r', encoding='utf-8') as ab:
            self.abv = ab.read()
        # self.abv = abbreviations
        self.infile = in_file

    def read_file(self, file="text.txt"):
        text = open(file, "r").read()
        return text

    def sentence_split(self, text):
        sent_pattern = r"(?<=[a-zäöü][.!?])\s+"
        sentences = re.split(sent_pattern, infile)
        return sentences

    def tokenize_sentences(self, sentence_list):
        new_sents = []
        for sent in sentences:
            for pos, char in enumerate(sent):
                if char in string.punctuation:
                    sent = sent.replace(char, " " + char + " ")
            new_sents.append(sent)

        marked_sentences = ["<s>" + sent + "</s>" + "\n" for sent in new_sents]
        for marked_sent in marked_sentences:
            print("|".join(marked_sent.split()))


tokenizer = HTMLTokenizer("abkuerzungen.txt", "text.txt")
infile = tokenizer.read_file()
sentences = tokenizer.sentence_split(infile)
print(tokenizer.tokenize_sentences(sentences))


