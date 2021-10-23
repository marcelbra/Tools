#!/usr/bin/env python

import re
import string
import sys


class HTMLTokenizer:

    def __init__(self, abbreviations, in_file):
        with open(abbreviations, 'r', encoding='utf-8') as ab:
            self.abv = ab.read()
        # self.abv = abbreviations
        self.infile = in_file

    def read_file(self, file="text.txt"):
        text = (open(file, "r")).read()
        return text

    def sentence_split(self, text):
        sent_pattern = r"(?<=[a-zäöü][.!?])\s+"
        sentences = re.split(sent_pattern, infile)
        return "\n".join(sentences)

    def tokenize_text(self):
        text = self.read_file()
        abbreviations = self.abv

        # print("\n".join(text.split(".")))

        tokenizer_pattern = r"([\w\s]+[\.\?\!]+)"

        # TODO: Satz endet immer nach Punkt außer bei Abkürzungen und zwischen Zahlen?
        #  An Fragezeichen und Ausrufezeichen denken!

        # print("\n".join(text.split(".")))

        tokenized = re.findall(tokenizer_pattern, text)
        # print(tokenized)


        # tokenized = [re.sub(r"[\":.;,]", '', word) for word in text.split()]

        for tok in tokenized:
            if tok in abbreviations:
                tok = tok.replace(tok, abbreviations)

        return tokenized

    def save_text(self, file_name, tokenized):
        with open(file_name, "w") as f:
            f.write("\n".join(tokenized))


tokenizer = HTMLTokenizer("abkuerzungen.txt", "text.txt")
infile = tokenizer.read_file()
text_tok = tokenizer.tokenize_text()
sentences = tokenizer.sentence_split(infile)
print(sentences)

tokenizer.save_text("text.tok", text_tok)













