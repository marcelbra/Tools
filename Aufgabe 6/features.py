"""
Extract features.
"""

import string

def word_tag(tag, words, ix):
    # Extract word-tag feature
    word, tag = (words[ix], tag)

    return word, tag


def prevtag_tag(prevtag, tag, ix):
    # extract previous tag for given index
    if ix == 0:
        prevtag, tag = (" ", tag)
    elif ix > 0:
        prevtag, tag = (prevtag, tag)

    return prevtag, tag


def prevtag_word_tag(prevtag, tag, words, ix):
    # extract previous tag, current word and current tag for given index
    if ix == 0:
        prev_tag, word, tag = (" ", words[ix], tag)
    elif ix > 0:
        prev_tag, tag, word = (prevtag, tag, words[ix])

    return prev_tag, tag, word


def substrings_tag(tag, words):
    # extract all substrings between length 3-6 for word at current index, tag at current index
    ngrams = []
    for word in words:
        for gram_size in range(3, 7):
            grams = [word[i:i + gram_size] for i in range(len(word) - gram_size + 1)]
            ngrams.extend((gram, tag) for gram in grams)

    return ngrams


def word_shape_tag(tag, words, ix):
    """Convert words to shapes depending on lowercase/ uppercase/ digits e.g. Testwort-717 -> Xx-0
    Return word shape, tag at current index"""
    word = words[ix]
    for char in word:
        if str.islower(char):
            word = word.replace(char, "x")
        elif str.isupper(char):
            word = word.replace(char, "X")
        elif str.isdigit(char):
            word = word.replace(char, "0")
        word = "".join(set(word))
    word_shape, tag = (word, tag)

    return word_shape, tag
