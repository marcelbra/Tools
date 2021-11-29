"""
Define feature extractors here.
"""

import math


def amount_exclamation_mark_pos(file, _class):
    return file.count("!") if _class == "spam" else 0

def amount_exclamation_mark_neg(file, _class):
    return file.count("!") if _class == "ham" else 0

def length_pos(file, _class):
    # Log operation dampens length and impedes overflow in exponential
    return math.log(len(file)) if _class == "spam" else 0

def length_neg(file, _class):
    return math.log(len(file)) if _class == "ham" else 0

def avg_word_length_pos(file, _class):
    return sum([len(word) for word in file]) / len(file) if _class == "spam" else 0

def avg_word_length_neg(file, _class):
    return sum([len(word) for word in file]) / len(file) if _class == "ham" else 0