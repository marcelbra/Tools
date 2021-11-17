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
        self.f_class = {cat: 0 for cat in text_categories}


    def count_freqs_from_file(self):
        f_word_class = {cat: {} for cat in self.text_categories}
        f_word = {}
        for root, dirs, files in os.walk(self.file_path):
            for cat in self.text_categories:
                if root.endswith(cat):
                    for file in files:
                        self.f_class[cat] += 1
                        email = open(os.path.join(root, file), encoding='latin-1').read()
                        for word in email.split():
                            if word in f_word_class[cat]:
                                f_word_class[cat][word] += 1
                            else:
                                f_word_class[cat][word] = 1
                        #f_word = {word:(1 if word not in f_word else f_word[word]+1) for word in email.split()}
                            if word in f_word:
                                f_word[word] += 1
                            else:
                                f_word[word] = 1
        print(f_word)
        return f_word_class, f_word


    def calculate_delta(self, f_word_class):
        N1 = sum(1 for cat in f_word_class for f in f_word_class[cat].values() if f == 1)
        N2 = sum(1 for cat in f_word_class for f in f_word_class[cat].values() if f == 2)

        delta = N1 / (N1 + 2 * N2)
        return delta


    def calculate_probs(self):
        """
        Method to calculate probabilities.
        Need to calculate:
        A) p(w)    from    f(w)/sum(f(all_words))
        B) p(c)    from    f(c)/sum(f(all_classes))
        C) p(w|c)  from    f(w,c)/sum(f(w', c))
        D) backoff factor alpha = 1-sum([f_w/f_all_w_in_c for w in c])
        E) discount delta: iterate over all pairs and their freq
        -> prob = (f-discount) / total
        """
        # A) and C)
        p_word = {}
        f_word_class, f_word = self.count_freqs_from_file()
        p_word_given_class = {cat: {} for cat in self.text_categories}
        f_allwords = sum([sum(words_freqs.values()) for _, words_freqs in f_word_class.items()])
        for cls, words_freqs in f_word_class.items(): # {cls: {wort:f}}
            f_allwords_in_cls = sum(words_freqs.values())
            for word, f_word_in_cls in words_freqs.items():            # f_word is freq word in class
                if word in p_word_given_class[cls]:
                    p_word_given_class[cls][word] += f_word_in_cls / f_allwords_in_cls
                else:
                    p_word_given_class[cls][word] = f_word_in_cls / f_allwords_in_cls
                if word in p_word:
                    p_word[word] += f_word[word] / f_allwords
                else:
                    p_word[word] = f_word[word] / f_allwords
        # B)
        p_class = [{cat: self.f_class[cat] / sum(self.f_class.values())} for cat in self.f_class]




nb = NaiveBayes("./train", ["ham", "spam"])
nb.calculate_probs()
