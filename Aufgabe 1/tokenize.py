#!/usr/bin/env python

import re
import string
import sys


class HTMLTokenizer:

    def __init__(self, abbreviations, in_file, out_file):
        with open(abbreviations+".txt", 'r', encoding='utf-8') as ab:
            self.abv = ab.read()
        self.infile = in_file
        self.outfile = out_file
        self.output = []

    def read_file(self):
        self.infile = open(self.infile, "r").read()
        return self.infile

    def sentence_split(self):
        sent_pattern = r"(?<=[a-zäöü][.!?])\s+"
        out = re.split(sent_pattern, self.read_file())
        return out

    def remove_pipe(self, idx1, idx2):
        self.output[idx1] = ""
        self.output[idx2] = ""
        return self.output

    def tokenize_sentences(self):
        new_sents = []
        for sent in self.sentence_split():
            for pos, char in enumerate(sent):
                if char in string.punctuation:
                    sent = sent.replace(char, " " + char + " ")
            new_sents.append(sent)
        marked_sentences = ["<s> " + sent + "</s>" + "\n" for sent in new_sents]
        for marked_sent in marked_sentences:
            self.output += ("|".join(marked_sent.split()))
        for pos, char in enumerate(self.output):
            if char == "|":
                # 1.) Check if (char_pos-1).isdigit() and (char_pos+3).isdigit()
                # Separated digits look like: 20|.|000
                # 2.) Check if ''.join((char_pos-1), (char_pos-2), (char_pos-3)) == "www"
                # if yes: remove (see in 1.))
                # 3.) Check if ''.join((char_pos+1), (char_pos+2)) == "de"
                # if yes: remove (see in 1.))
                # Separated addresses look like "www|.|google|.|de"
                try:
                    if self.output[pos-1].isdigit() and self.output[pos+3].isdigit():
                        self.remove_pipe(pos, pos+2)
                    if self.output[pos-1] + self.output[pos-2] + self.output[pos-3] == "www":
                        self.remove_pipe(pos, pos+2)
                    if self.output[pos+1] + self.output[pos+2] == "de":
                        self.remove_pipe(pos-2, pos)
                except IndexError:
                    # IndexError only occurs when a "|" is being checked that is at the end of sentence.
                    # This case is redundant to the task anyway, so we can just pass this error.
                    pass
        self.output = ''.join(self.output)
        self.output = self.output.replace("|", " ")
        return self.output

    def save_text(self):
        self.tokenize_sentences()
        with open(self.outfile, "w") as f:
            f.write('</s>\n'.join(str(self.output).split("</s>")))


tokenizer = HTMLTokenizer(sys.argv[1], sys.argv[2], "text.tok")
tokenizer.save_text()



