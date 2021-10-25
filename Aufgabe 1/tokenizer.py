#!/usr/bin/env python

import re
import sys
import string

class RawTextTokenizer:

    def __init__(self):
        """ #
        Expects a corpus of running text.
        Splits senteces greedily by looking for punctuation like . ! ? and ." .
        Makes sure abbreviations, thousands-denoted numbers and internet addresses are not split.
        The idea is: replace the dots of these three classes with a unique token.
        Then perform the greedy splitting. Then replace the this unique token back into a dot.

        How to run:
        $ python3 tokenizer.py abbreviations text.txt > text.tok
        """
        assert len(sys.argv) > 0, "You need to specify as 'abrev infile.txt > outfile.txt"
        self.abrev_path = sys.argv[1] + ".txt"
        self.in_file_path = sys.argv[2]
        self.abrev, self.text, self.sentences = "", "", ""
        self.tokenized = []

    def load_data(self):
        """
        Loads abbreviations and input file.
        """
        with open(self.abrev_path, "r", encoding='utf-8') as f:
            self.abrev = set(f.read().split("\n"))
        with open(self.in_file_path, "r", encoding='utf-8') as f:
            self.text = f.read()

    def split_sentences(self):
        """
        Replaces all dots which are not grammatical dots with "QU4K". This includes
        1) words which are in the abbreviations list, 2) numbers and 3) internet addresses.
        Next, 4) Split by . , ! ? and 5) replace QU4K back to dot.
        This results in correct split since dots which are not grammatical ones anymore
        are not considered for splitting.
        """
        text = " ".join([word.replace(".", "QU4K") if word in self.abrev else word  # 1)
                         for i, word in enumerate(self.text.split())])
        number_pattern = r"(?<!\S)\d{1,3}(?:\.\d{3})*(?!\S)"
        text = re.sub(number_pattern, lambda m: m.group().replace('.', 'QU4K'), text)  # 2)
        address_pattern = r"(www).([A-Za-z0-9]*)\.(de|com|org)"
        re.sub(address_pattern, r"\1QU4K\2QU4K\3", text)  # 3)
        split_pattern = r"((?<=\.\")|(?<=[.!?]))\s+"
        self.sentences = re.split(split_pattern, text)  # 4)

    def tokenize(self):
        """Tokenizes sentences into single words.
        Adds a space in front or after a punctuation character in greedy fashion.
        Brute force, but linear in characters."""
        #split_by = r"[!%&'\(\)$#\"\/\\*+,-.:;<=>?@\[\]^_´`{|}~]"
        space = set("[!%&'()$#\"/\*+,-.:;<=>?@[]^_´`{|}~]")
        for sent in self.sentences:
            sent = list(sent)
            for i, char in enumerate(sent):
                # Make sure to respect length of string when indexing
                if i != 0:
                    # insert space in front if char is punctuation
                    if sent[i] in space and sent[i - 1] != " ":
                        sent.insert(i, " ")
                if i != len(sent)-1:
                    # insert space after if char is punctuation
                    if sent[i] in space and sent[i + 1] != " ":
                        sent.insert(i + 1, " ")
            sent = self.reset_dots("".join(sent))
            self.tokenized.append(sent)

    def reset_dots(self, sentence):
        """
        Replaces the unique token QU4K with a dot.
        """
        return sentence.replace("QU4K", ".")

    def print_text(self):
        """
        Saves the text to the specified output file.
        """
        s = ""
        for sentence in self.tokenized:
            s += f"{sentence}\n"
            print(sentence)

def main():
    tokenizer = RawTextTokenizer()
    tokenizer.load_data()
    tokenizer.split_sentences()
    tokenizer.tokenize()
    tokenizer.print_text()

main()
