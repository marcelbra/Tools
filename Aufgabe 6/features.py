import string

"""Define extracted features"""


def word_tag(sentence_tags_list):
    # Extract word-tag feature
    words_tags = []
    for (words, tags) in sentence_tags_list:
        word_tag = list(zip(words, tags))
        words_tags.append(word_tag)
    return words_tags


def word_tag_start(sentence_tags_list):
    # differentiate between word_tag tuple at the beginning of a sentence vs. not at the beginning
    start_other_tag = []
    for (words, tags) in sentence_tags_list:
        word_tag_list = list(zip(words, tags))
        start_other_tag.append(str(word_tag_list[0]) + "," + "start")
        for el in word_tag_list[1:]:
            start_other_tag.append(str(el) + "," + "not_start")
    return start_other_tag


def word_shape_tag(sentence_tags_list):
    """Convert words to shapes depending on lowercase/ uppercase/ digits e.g. Testwort-717 -> Xx-0"""
    shape_tags = []
    for (words, tags) in sentence_tags_list:
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
        shape_tag = list(zip(new_words, tags))
        shape_tags.append(shape_tag)
    return shape_tags

