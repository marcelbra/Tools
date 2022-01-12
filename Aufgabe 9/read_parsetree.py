"""
P7 Tools - Aufgabe 9
Parser (Preprocessing)

Gruppe:
Marcel Braasch
Nadja Seeberg
Sinem Kühlewind (geb. Demiraslan)
"""

from collections import defaultdict
import re
from utils import is_letter, split_sentence_at_brackets, extract_word




class Parser:
    def __init__(self):
        self.path = "data/train.txt"


    def get_sentences(self):
        with open(self.path, encoding="utf-8") as f:
            text = f.read().split("\n")
        for sentence in text:
            yield sentence


    def process_sentence(self):
        """
        Processes parsetree of sentences in data set.
        Calls do_parse() method.
        """
        sentences = list(self.get_sentences())[1:4:2]   # two simple sample sentences
        for sent in sentences:
            tmp_list = []
            self.do_parse(sentence=sent, index=0, tmp=tmp_list)
        return 0


    def do_parse(self, sentence, index: int, tmp):
        """
        @return: words, constituents, indexes
        """
        sentence_list = split_sentence_at_brackets(sentence=sentence)
        preterminal_to_word = dict()
        parent_to_child = dict()
        for current_pos, char in enumerate(sentence[index:]):
            if char == "(":
                return self.do_parse(sentence, index+(1 if current_pos == 0 else current_pos), tmp)
            elif char == " ":
                remaining_sentence_list = split_sentence_at_brackets(sentence=sentence[index:])
                preterminal_to_word[remaining_sentence_list[0]] = remaining_sentence_list[1]
                tmp.append(preterminal_to_word)
        words = [val for dict_ in tmp for key, val in dict_.items()]
        #for list of constituents, do enumerate(words)
        # return words


parser = Parser()
parser.process_sentence()


