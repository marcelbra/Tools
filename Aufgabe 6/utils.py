"""
Helper file for various operations.
"""

from collections import Counter
from typing import List, Union
import math
import re

Numeric = Union[int, float]
Vector = List[Numeric]


def add(list_1: Vector,
        list_2: Vector
        ) -> Vector:
    return [x + list_2[i] for i, x in enumerate(list_1)]


def sub(list_1: Vector,
        list_2: Vector
        ) -> Vector:
    return [x - list_2[i] for i, x in enumerate(list_1)]


def mul(list_1: Vector,
        list_2: Vector
        ) -> Vector:
    return [x * list_2[i] for i, x in enumerate(list_1)]


def div(list_1: Vector,
        list_2: Vector
        ) -> Vector:
    return [x / list_2[i] for i, x in enumerate(list_1)]


def dot(list_1: Vector,
        list_2: Vector
        ) -> Numeric:
    return sum(mul(list_1, list_2))


def create_vec(value: Numeric,
               n: Numeric
               ) -> Vector:
    return [value] * n


def log_sum_exp(a: Numeric,
                b: Numeric
                ) -> Numeric:
    if b > a: a, b = b, a
    return a + math.log(1 + math.exp(b - a))


def get_substrings_tag(tag, word, start=3, end=7):
    word = " " + word + " "
    return [(word[i:i + l], tag) for l in range(start, end + 1)
            for i in range(len(word) - l + 1)]


def sign(v: List):
    sign_vector = []
    for x in v:
        if x == 0:
            sign_vector.append(0)
        elif x > 1:
            sign_vector.append(1)
        elif x < 1:
            sign_vector.append(-1)
    return sign_vector


def get_word_shape(word):
    """Convert words to shapes depending on lowercase/ uppercase/ digits e.g. Testwort-717 -> Xx-0
    Return word shape, tag at current index"""
    word = re.sub(r"[a-zäüöß]+", "x", word)
    word = re.sub(r"[A-ZÄÖÜ]+", "X", word)
    word = re.sub(r"\d+", "0", word)
    return word


def get_lexical_features(tag, words, i):
    word_tag = "WT", words[i], tag
    word_shape_tag = "ShT", get_word_shape(words[i]), tag
    ngrams_tag = get_substrings_tag(tag, words[i])
    return [word_tag, word_shape_tag] + ngrams_tag


def get_context_features(prevtag, tag, words, i):
    prevtag_tag = prevtag, tag
    prevtag_word_tag = prevtag, tag, words[i]
    return [prevtag_tag, prevtag_word_tag]


def feature_extraction(adjacent_tag, tag, words, i):
    lexical = get_lexical_features(tag, words, i)
    # when we are in backward adjacent_tag will be tag and vice versa
    context = get_context_features(adjacent_tag, tag, words, i)
    features = lexical + context
    feature_count = {str(k): v for k, v in Counter(features).items()}
    return feature_count


def init_scores(words, forward, tagset):
    table = [{tag: 0 for tag in tagset} for _ in words]
    i, tag = (0, "<s>") if forward else (-1, "</s>")
    table[i][tag] = 1.0
    return table
