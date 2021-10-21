import re
import sys


class HTMLTokenizer:

    def __init__(self, abbreviations, in_file):
        with open(abbreviations, 'r', encoding='utf-8') as ab:
            self.abv = ab.read()
        #self.abv = abbreviations
        self.infile = in_file


    def read_file(self, file="text.txt"):
        text = (open(file, "r")).read()
        return text


    def tokenize_text(self):
        text = self.read_file()
        abbreviations = self.abv

        specialchars = ["\"", ":", ",", ";"]
        for c in specialchars:
            if c in text:
                text = text.replace(c, " " + c + " ")

        tokenized = [word for word in text.split()]
        for tok in tokenized:
            if tok in abbreviations:
                tok = tok.replace(tok, abbreviations)

        return tokenized



    def save_text(self, file_name, tokenized):
        with open(file_name, "w") as f:
            f.write("\n".join(tokenized))




tokenizer = HTMLTokenizer(sys.argv[1], sys.argv[2])
infile = tokenizer.read_file()
text_tok = tokenizer.tokenize_text()

tokenizer.save_text("text.tok", text_tok)