# !/usr/bin/python

import sys
import os


class NaiveBayes:
    def __init__(self, file_path, text_categories):
        self.file_path = file_path
        self.text_categories = text_categories
        self.word_class_freq = {}
        self.class_freq = {cat: 0 for cat in text_categories}

        for cat in text_categories:
            if cat not in self.word_class_freq:
                self.word_class_freq[cat] = {}

    def read_files(self):
        for root, dirs, files in os.walk(self.file_path):
            for cat in self.text_categories:
                if root.endswith(cat):
                    for file in files:
                        self.class_freq[cat] += 1
                        email = open(os.path.join(root, file), encoding='latin-1').read()
                        for word in email.split():
                            if word in self.word_class_freq[cat]:
                                self.word_class_freq[cat][word] += 1
                            else:
                                self.word_class_freq[cat][word] = 1


nb = NaiveBayes("./train", ["ham", "spam"])
nb.read_files()
print(nb.class_freq)
print(nb.word_class_freq.keys())
