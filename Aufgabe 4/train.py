# !/usr/bin/python

import sys
import os


class NaiveBayes:
    def __init__(self, file_path, text_categories):
        """
        :text_categories: available classes; one class equals one subdirectory
        :word_class_freq: nested dictionary; {class: {word: freq}, ...}
        -> p(w|c) = f(w,c)/sum([1 for w in c]))
        :class_freq: stands for f(c)
        -> p(c) = f(c)/sum([1 for c in classes])
        """
        self.file_path = file_path
        self.text_categories = text_categories
        self.texts = []
        self.f_class = {cat: 0 for cat in text_categories}


    def read_files(self):
        for root, dirs, files in os.walk(self.file_path):
            for cat in self.text_categories:
                if root.endswith(cat):
                    for file in files:
                        self.f_class[cat] += 1
                        self.texts.append(open(os.path.join(root, file), encoding='latin-1').read())
        return self.texts


    def count_frequencies(self):
        self.read_files()
        f_word_class = {cat:{} for cat in self.text_categories}
        for cat in self.text_categories:
            for mail in self.texts:
                for word in mail.split():
                    if word in f_word_class[cat]:
                        f_word_class[cat][word] += 1
                    else:
                        f_word_class[cat][word] = 1
        return f_word_class


    def calculate_probs(self):
        """
        Method to calculate probabilities.
        Need to calculate:
        A) p(w)    from    f(w)/sum(f(all_words))
        B) p(c)    from    f(c)/sum(f(all_classes))
        C) p(w|c)  from    f(w,c)/sum(f(w', c))
        D) backoff factor alpha
        E) discount delta: iterate over all pairs and their freq
        """
        # A) and C)
        p_word = {}
        f_word_class = self.count_frequencies()
        p_word_given_class = {cat: {} for cat in self.text_categories}
        f_allwords = sum([sum(words_freqs.values()) for _, words_freqs in f_word_class.items()])
        for cls, words_freqs in f_word_class.items():
            f_allwords_in_cls = sum(words_freqs.values())
            for word, f_word in words_freqs.items():
                if word in p_word_given_class[cls]:
                    p_word_given_class[cls][word] += f_word / f_allwords_in_cls
                else:
                    p_word_given_class[cls][word] = f_word / f_allwords_in_cls
                if word in p_word:
                    p_word[word] += f_word / f_allwords
                else:
                    p_word[word] = f_word / f_allwords
        # B)
        p_class = [{cat: self.f_class[cat] / sum(self.f_class.values())} for cat in self.f_class]




nb = NaiveBayes("./train", ["ham", "spam"])
nb.calculate_probs()
