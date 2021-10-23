#!/usr/bin/env python

import re
import string
import sys


class HTMLTokenizer:

    def __init__(self, abbreviations, in_file, out_file):
        with open(abbreviations, 'r', encoding='utf-8') as ab:
            self.abv = ab.read()
        self.infile = in_file
        self.outfile = out_file

    def read_file(self):
        text = (open(self.infile, "r")).read()
        return text

    def sentence_split(self):
        sent_pattern = r"(?<=[a-zäöü][.!?])\s+"
        sentences = re.split(sent_pattern, self.read_file())
        return "\n".join(sentences)

    def tokenize_text(self):
        text = self.read_file()
        tokenizer_pattern = r"([\w\s]+[\.\?\!]+)"

        # TODO: Satz endet immer nach Punkt außer bei Abkürzungen und zwischen Zahlen?
        #  An Fragezeichen und Ausrufezeichen denken!

        tokenized = re.findall(tokenizer_pattern, text)

        for tok in tokenized:
            for a in self.abv:
                if tok == a:
                    tok = tok.replace(tok, self.abv)

        return tokenized

    def save_text(self, sentences):
        with open(self.outfile, "w") as f:
            f.write(sentences)


tokenizer = HTMLTokenizer("abbreviations.txt", "text.txt", "text.tok")
text_tok = tokenizer.tokenize_text()
sentences = tokenizer.sentence_split()
#print(sentences)

tokenizer.save_text(sentences)













