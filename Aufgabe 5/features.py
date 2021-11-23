"""
Define feature extractors here.
"""

def amount_exclamation_mark_of(file, _class):
    return file.count("!") if _class == "ham" else 0

def length_of(file, _class):
    return len(file) if _class == "spam" else 0

def avg_word_length_of(file, _class):
    return sum([len(word) for word in file]) / len(file) if _class == "spam" else 0