import string

"""Extract features"""


def word_tag(words, tags):
    # Extract word-tag feature
    words_tags = list(zip(words, tags))
    return words_tags


def prevtags_tags(words, tags):
    # extract tag and previous tag for each tag - previous tag for first tag is <s>
    prevtags_tags = []
    for ix, tag in enumerate(tags):
        prevtag_tag = []
        if ix == 0:
            prevtag_tag.append((tag, "<s>"))
        if ix > 0:
            prev_tag = tags[ix - 1]
            prevtag_tag.append((tag, prev_tag))
        prevtags_tags.append(prevtag_tag)
    return prevtags_tags


def prevtag_word_tag(words, tags):
    # extract previous tag, word and tag for given tag
    prevtags_words_tags = []

    for ix, tag in enumerate(tags):
        prevtag_word_tag = []
        if ix == 0:
            prevtag_word_tag.append(("<s>", words[0], tags[0]))
        elif ix > 0:
            prev_tag = tags[ix - 1]
            tag = tags[ix]
            word = words[ix]
            prevtag_word_tag.append((prev_tag, word, tag))
        prevtags_words_tags.append(prevtag_word_tag)
    return prevtags_words_tags


def substrings_tag(words, tags):
    # all Substrings between length 3-6, tag
    ngrams_tags = []
    for ix, word in enumerate(words):
        ngrams_tag = []
        ngrams = []
        for gram_size in range(3, 7):
            grams = [word[i:i + gram_size] for i in range(len(word) - gram_size + 1)]
            ngrams.extend(grams)
        ngrams_tag.append((ngrams, tags[ix]))
        ngrams_tags.append(ngrams_tag)
    return ngrams_tags


def word_shape_tag(words, tags):
    """Convert words to shapes depending on lowercase/ uppercase/ digits e.g. Testwort-717 -> Xx-0"""
    new_words = []
    for word in words:
        for char in word:
            if str.islower(char):
                word = word.replace(char, "x")
            elif str.isupper(char):
                word = word.replace(char, "X")
            elif str.isdigit(char):
                word = word.replace(char, "0")
            word = "".join(set(word))
        new_words.append(word)
    shapes_tags = list(zip(new_words, tags))

    return shapes_tags
