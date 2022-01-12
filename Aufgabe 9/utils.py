import re


def is_letter(char: str):
    letter = r"\w"
    match = re.findall(letter, char)
    if match:
        return True


def split_sentence_at_brackets(sentence: str):
    split_conditions = "( )".split()
    for char in sentence:
        if char in split_conditions:
            sentence = sentence.replace(char, " ")
    return sentence.split()


def extract_word(sentence: str, position: int):
    sent = split_sentence_at_brackets(sentence[position:])
    return sent[0]



# TODO function to summarize chain rules