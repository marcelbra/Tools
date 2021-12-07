"""
Extract features.
"""

import string

def word_tag(ix, words, tags):
    # Extract word-tag feature
    word, tag = (words[ix], tags[ix])

    return word, tag


def prevtag_tag(ix, tags):
    # extract previous tag for given index
    if ix == 0:
        prevtag, tag = (" ", tags[ix])
    elif ix > 0:
        prevtag, tag = (tags[ix-1], tags[ix])

    return prevtag, tag


def prevtag_word_tag(ix, words, tags):
    # extract previous tag, current word and current tag for given index
    if ix == 0:
        prev_tag, word, tag = (" ", words[ix], tags[ix])
    elif ix > 0:
        prev_tag, tag, word = (tags[ix - 1], tags[ix], words[ix])

    return prev_tag, tag, word


def substrings_tag(ix, words, tags):
    # extract all substrings between length 3-6 for word at current index, tag at current index
    ngrams_tag = []
    ngrams = []
    for gram_size in range(3, 7):
        grams = [words[ix][i:i + gram_size] for i in range(len(words[ix]) - gram_size + 1)]
        ngrams.extend(grams)
    ngrams_tag.append((ngrams, tags[ix]))

    return ngrams_tag


def word_shape_tag(ix, words, tags):
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
    word_shape, tag = (word, tags[ix])

    return word_shape, tag
