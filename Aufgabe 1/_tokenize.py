#!/usr/bin/env python

import re
import sys
import string

class HTMLTokenizer:

    def __init__(self):
        assert len(sys.argv) > 0, "You need to specify as 'abrev infile.txt > outfile.txt"
        self.abrev_path = sys.argv[1] + ".txt"
        self.in_file_path = sys.argv[2]
        self.out_file_path = sys.argv[4]
        self.abrev, self.text, self.sentences = "", "", ""
        self.tokenized = []

    def load_data(self):
        """Loads abbrev
        iations and input file."""
        with open(self.abrev_path, "r", encoding='utf-8') as f:
            self.abrev = set(f.read().split("\n"))
        with open(self.in_file_path, "r", encoding='utf-8') as f:
            self.text = f.read()

    def split_sentences(self):
        """
        Replaces all dots which are not grammatical dots with "QU4K". This includes 1) words which
        are in the abbreviations list, 2) numbers and 3) internet addresses. Next, 4) Split by . , ! ? and
        5) replace QU4K back to dot. This results in correct split since dots which are not grammatical
        ones anymore are not considered for splitting.
        """
        text = "Es ist z.B. schön, dass ca. 1.000.000 andere Kinder (in Nepal und China) beim Dr. Arzt Abk. und Abl. nicht unterscheiden können."
        text += " Außerdem regnet es in München."
        text = " ".join([word.replace(".", "QU4K") if word in self.abrev else word  # 1)
                         for i, word in enumerate(text.split())])
        number_pattern = r"(?<!\S)\d{1,3}(?:\.\d{3})*(?!\S)"
        text = re.sub(number_pattern, lambda m: m.group().replace('.', 'QU4K'), text)  # 2)
        address_pattern = r"(www).([A-Za-z0-9]*)\.(de|com|org)"
        re.sub(address_pattern, r"\1QU4K\2QU4K\3", text)  # 3)
        split_pattern = r"(?<=[a-zäöü][.!?])\s+"
        self.sentences = re.split(split_pattern, text)  # 4)

    def tokenize(self):
        """Tokenizes sentences into singlue words."""
        #split_by = r"[!%&'\(\)$#\"\/\\*+,-.:;<=>?@\[\]^_´`{|}~]"
        space = set("[!%&'()$#\"/\*+,-.:;<=>?@[]^_´`{|}~]")
        for sent in self.sentences:
            sent = list(sent)
            for i, char in enumerate(sent):
                # Make sure to respect length of string when indexing
                if i != 0:
                    if sent[i] in space and sent[i - 1] != " ":
                        sent.insert(i, " ")
                if i != len(sent)-1:
                    if sent[i] in space and sent[i + 1] != " ":
                        sent.insert(i + 1, " ")
            sent = self.reset_dots("".join(sent))
            self.tokenized.append(sent)

    def reset_dots(self, sentence):
        return sentence.replace("QU4K", ".")

    def save_text(self):
        """Saves the text to the specified output file."""
        s = ""
        for sentence in self.tokenized:
            s += f"{sentence}\n"
        with open(self.out_file_path, "w") as f:
            f.write(s)


tokenizer = HTMLTokenizer()
tokenizer.load_data()
tokenizer.split_sentences()
tokenizer.tokenize()
tokenizer.save_text()



