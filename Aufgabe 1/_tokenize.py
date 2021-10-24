#!/usr/bin/env python

import re
import sys


class HTMLTokenizer:

    def __init__(self):
        assert len(sys.argv) > 0, "You need to specify as 'abrev infile.txt > outfile.txt"
        self.abrev_path = sys.argv[1] + ".txt"
        self.in_file_path = sys.argv[2]
        self.out_file_path = sys.argv[4]
        self.abrev, self.text, self.sentences = "", "", ""


    def load_data(self):
        with open(self.abrev_path, "r", encoding='utf-8') as f:
            self.abrev = set(f.read().split("\n"))

        with open(self.in_file_path, "r", encoding='utf-8') as f:
            self.text = f.read()

    def split_sentences(self):
        """
        Replaces all dots which are not grammatical dots with "QU4K".
        Then performs splitting by dots.
        Sentences with e.g., "Dr.", "ca.", etc. will not be split because they are now "DrQU4K", "caQU4K"
        Replace back the QU4Ks to dots.
        :return:
        """
        text = "Es ist z.B. schön, dass ca. 1.000.000 andere der Kidner zw. Nepal und China beim Dr. Arzt lachen. Zudem kann man sagen es ist warm. Nicht wirklich kalt."
        # text = self.text

        # QU4K abbreviations
        text = " ".join([word.replace(".", "QU4K") if word in self.abrev else word
                         for i, word in enumerate(text.split())])
        # QU4K numbers
        number_pattern = r"(?<!\S)\d{1,3}(?:\.\d{3})*(?!\S)"
        text = re.sub(number_pattern, lambda m: m.group().replace('.', 'QU4K'), text)

        # QU4K internet addresses
        address_pattern = r"(www).([A-Za-z0-9]*)\.(de|com|org)"
        re.sub(address_pattern, r"\1QU4K\2QU4K\3", text)

        # Split by . , ! ?
        # Dots which are not denoting sentence end are now QU4Ks!
        split_pattern = r"(?<=[a-zäöü][.!?])\s+"
        sentences = re.split(split_pattern, text)

        # Turn QU4Ks back into dots
        sentences = [sentence.replace("QU4K", ".") for sentence in sentences]
        self.sentences = sentences

    def tokenize(self):
        self.tokenized = [sent.split(" ") for sent in self.sentences]

    def save_text(self):
        s = ""
        for sentence in self.sentences:
            s += f"{sentence}\n"
        with open(self.out_file_path, "w") as f:
            f.write(s)


tokenizer = HTMLTokenizer()
tokenizer.load_data()
tokenizer.split_sentences()
#tokenizer.tokenize()
tokenizer.save_text()



