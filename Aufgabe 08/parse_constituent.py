import os
import sys
import re
# from tqdm import tqdm
import pickle
from tag_helper import map_tag_to_modified_tag, modified_tag_to_tag
from tree import Tree

class DataCreator:

    def __init__(self, data_path, tagset_path):
        self.tags = self.load_tagset(tagset_path)
        self.trees = self.load_data(data_path)

    def run(self):
        data = {}
        new_trees = []
        for name, trees in self.trees.items():
            current = []
            for i, parse_tree in enumerate(trees):
                new_trees.append(parse_tree[:15])

                # Debugging variables
                # s = 0
                # self.parse_tree = parse_tree
                # self.i = i

                # Variables things will be saved to
                self.words = []
                self.constituents = []
                self.index = 0

                # Create tokens
                self.tokens = [x[0] for x in re.finditer(r"[()]|[^\s()]+", parse_tree.replace("\n", ""))][1:]
                self.tokens = map_tag_to_modified_tag(self.tokens)  # Maps tags like "."  to ".-TAG"

                # Constructs tree representation, word and constituent list
                tree = self.construct_tree()
                self.construct_constituents(tree)

                original = self.reconstruct_original()

                current.append({"consituents": self.constituents,
                                "words": self.words})
            data[name] = current

        return data

    def construct_tree(self):
        while True:
            token = self.tokens.pop(0)
            if token == ")":
                return node
            elif token == "(":
                child = self.construct_tree()
                has_other_children = node.children == []
                node.children.append(child)
                next_token = self.tokens[0]
                # Get current node's child indices to build its indices
                indices = []
                for child in node.children:
                    indices.extend(child.get_indices())
                node.start = min(indices)
                node.end = max(indices)
                # Merging
                if next_token != "(" and has_other_children:
                    node_name = node.name  #
                    child = node.children[0]
                    child_name = child.name
                    node = child
                    node.name = f"{node_name}={child_name}"
                    s = 0
            elif token in self.tags:
                node = Tree(token)
            else:
                self.words.append(token)
                try:
                    node.start = self.index
                except:
                    print(token)
                    print(self.i)
                    sys.exit()
                node.end = self.index + 1
                self.index += 1

    def generate_tagset(self, path, save_tagset=False):
        tags = set()
        names = os.listdir(path)
        modified = tag_to_modified_tag()
        for name in names:
            with open(path + name, "r") as f:
                func = lambda tree: [x for x in re.finditer(r"[()]|[^\s()]+", tree)]
                for line in tqdm(f):
                    tokens = [x[0] for x in func(line)]
                    for i in range(1, len(tokens)):
                        if tokens[i - 1] == "(":
                            tags.add(tokens[i])
        tags = [modified[tag] for tag in tags]
        if save_tagset:
            with open("tagset.pkl", "wb") as (handle):
                pickle.dump(tags, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return tags

    def load_tagset(self, path):
        try:
            tags = []
            with open(path, "rb") as handle:
                tags = pickle.load(handle)
            return tags
        except:
            return self.generate_tagset(path)

    def construct_constituents(self, node):
        """Do in-order traversal to construct the constituent list."""
        self.constituents.append((node.name, node.start, node.end))
        for child in node.children:
            self.construct_constituents(child)

    def load_data(self, path):
        names = os.listdir(path)
        # counter = 0
        trees = {}
        for name in names:
            with open(path + name) as f:
                current = []
                for line in f:
                    current.append(line)
                trees[name.split(".")[0]] = current
        return trees

    def reconstruct_original(self):
        closing = []
        tree = "("
        for constituent in self.constituents:

            name = modified_tag_to_tag(constituent[0])
            start, end = constituent[1], constituent[2]
            diff = end - start

            # Restore merged constituents and set flag for adding two brackets
            add_two_brackets = True if "=" in name else False
            name = name.replace("=", "(")

            # Add constituent
            tree += name if tree[-1] == "(" else "(" + name

            # If we have reached a leaf add the next word
            if diff == 1:
                """Die Rekonstruktion ist noch etwas fehlerhaft.
                Hier an der Stelle kann es passieren, dass wir trotz einer
                Differenz von 1 keine Wörter mehr haben. Zeitlich bedingt
                haben wir es nicht geschafft diesen Bug zu beheben."""
                if self.words:
                    tree += " " + self.words.pop(0) + ")"
                # Decrement bracket counter once we have seen a word
                closing = [x - 1 for x in closing]

            # Add closing brackets
            if diff != 1:
                closing.append(diff)
            if add_two_brackets:
                if not tree.startswith('(TOP('):  # Special case in the beginning
                    closing.append(diff - 1)
            tree += ")" * closing.count(0)
            # Remove 0s we just used
            closing = [x for x in closing if x]

        tree += ")"
        return tree

def main():
    data_path = "/Aufgabe 08/PennTreebank/"
    tagset_path = "/Aufgabe 08/tagset.pkl"
    creator = DataCreator(data_path, tagset_path)
    data = creator.run()

if __name__ == "__main__":
    main()
