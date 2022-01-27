class Parser:
    def __init__(self):
        self.file_path = "./data/train.txt"

    def read_data(self, file):
        with open(file) as f:
            data = f.readlines()
            for sent in data[:1]:
                all_constituents = []
                for ix, char in enumerate(sent):
                    const = ""
                    if ix < len(sent) - 1:
                        all_constituents.append(
                            self.process_tree(sent, ix, const, constituents=[]))

        return all_constituents

    def process_tree(self, sentence, current_pos, const, constituents):
        # start_ix = current_pos
        # end_ix = 0
        words = []
        brackets = ["(", ")"]
        current_char = sentence[current_pos]
        next_pos = current_pos + 1
        next_char = sentence[next_pos]

        if current_char != ")":
            if current_char == "(":
                const = "" + next_char
                current_pos = next_pos
            elif current_char not in brackets and next_char != "(" and next_char != " ":
                const = const + next_char
                current_pos = next_pos
            elif current_char not in brackets and next_char == "(":
                constituents.append(const)
                current_pos = next_pos
            elif current_char not in brackets and next_char == " ":
                constituents.append(const)
                current_pos = next_pos

                words.append(self.extract_word(sentence, current_pos, word=""))

            self.parse_sentence(sentence, current_pos, const, constituents)

        else:
            return constituents, words

    def extract_word(self, sentence, current_pos, word):

        while sentence[current_pos] != ")":
            current_pos = current_pos + 1
            word = word + sentence[current_pos]
            self.extract_word(sentence, current_pos, word)
        else:
            return word


if __name__ == '__main__':
    parser = Parser()
    print(parser.read_data(parser.file_path))
