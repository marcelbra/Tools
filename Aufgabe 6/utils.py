"""
Helper file for various operations.
"""

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
        )-> Vector:
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
    return a + math.log(1+math.exp(b-a))

def get_substrings_tag(tag, words, start=3, end=7):
    """
    Extract all substrings between length 3-6 for word at current index, tag at current index
    """
    grams = []
    for word in words:
        if len(word) >= start:
            for size in range(start, end):
                for i in range(len(word) - size + 1):
                    grams.append((word[i:i + size], tag))
    return grams

def get_word_shape(word):
    """Convert words to shapes depending on lowercase/ uppercase/ digits e.g. Testwort-717 -> Xx-0
    Return word shape, tag at current index"""
    try:
        word = re.sub(r"[a-zäüöß]+", "x", word)
    except:
        print(word, type(word))
    word = re.sub(r"[A-ZÄÖÜ]+", "X", word)
    word = re.sub(r"\d+", "0", word)
    return word