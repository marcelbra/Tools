import os
import re
from tqdm import tqdm
import pickle
from tag_helper import (map_tag_to_word,
                        map_word_to_tag,
                        restore_replaced_tag,
                        substitute_tags)
class Tree:

    def __init__(self, name):
        self.children = []
        self.name = name
        self.start = 0
        self.end = 0

    def get_indices(self):
        return [self.start, self.end]

def generate_tagset(path, save_tagset=False):
    tags = set()
    names = os.listdir(path)
    for name in names:
        with open(path + name, "r") as f:
            func = lambda tree: [x for x in re.finditer(r"[()]|[^\s()]+", tree)]
            for line in tqdm(f):
                tokens = [x[0] for x in func(line)]
                for i in range(1, len(tokens)):
                    if tokens[i - 1] == "(":
                        tags.add(tokens[i])
    tags = [map_word_to_tag(x) for x in tags]
    if save_tagset:
        with open("tagset.pkl", "wb") as (handle):
            pickle.dump(tags, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return tags

def load_tagset():
    path = "/home/marcelbraasch/PycharmProjects/Tools/Aufgabe 8/tagset.pkl"
    tags = []
    with open(path, "rb") as handle:
        tags = pickle.load(handle)
    return tags

def construct_tree():
    global index
    while True:
        token = tokens.pop(0)
        if token == ")":
            return node
        elif token == "(":
            child = construct_tree()
            has_other_children = node.children == []
            node.children.append(child)
            next_token = tokens[0]
            # Get current node's child indices to build its indices
            indices = []
            for child in node.children:
                indices.extend(child.get_indices())
            node.start = min(indices)
            node.end = max(indices)
            # Merging
            if next_token != "(" and has_other_children:
                node_name = node.name #
                child = node.children[0]
                child_name = child.name
                node = child
                node.name = f"{node_name}={child_name}"
                s = 0
        elif token in tags:
            node = Tree(token)
        else:
            words.append(token)
            node.start = index
            node.end = index + 1
            index += 1

def reconstruct_original():
    closing = []
    tree = "("
    for constituent in constituents:
        constituent = restore_replaced_tag(constituent)
        start, end = constituent[1], constituent[2]
        diff = end - start

        # Restore merged constituents and set flag for adding two brackets
        add_two_brackets = True if "=" in constituent[0] else False
        name = constituent[0].replace("=", "(")
        # Add constituent
        tree += name if tree[-1] == "(" else "(" + name

        # If we have reached a leaf ad d the next word
        if diff == 1:
            tree += " " + words.pop(0) + ")"
            # Decrement bracket counter once we have seen a word
            closing = [x - 1 for x in closing]

        # Add closing brackets
        if diff != 1:
            closing.append(diff)
        if add_two_brackets:
            if tree != '(TOP(S':  # Special case in the beginning
                closing.append(diff - 1)
        tree += ")" * closing.count(0)
        # Remove 0s we just used
        closing = [x for x in closing if x]

    tree += ")"
    return tree

def construct_constituents(node):
    """Do in-order traversal to construct the constituent list."""
    constituents.append((node.name, node.start, node.end))
    for child in node.children:
        construct_constituents(child)

def load_test_data():
    path = "/home/marcelbraasch/PycharmProjects/Tools/Aufgabe 8/PennTreebank/"
    name = os.listdir(path)[0]
    counter = 0
    trees = []
    with open(path + name) as f:
        for line in f:
            trees.append(line)
            counter += 1
            if counter >= 10:
                break
    return trees

tags = load_tagset()
trees = load_test_data()

# A run for one example
parse_tree = trees[0]
tokens = substitute_tags([x[0] for x in re.finditer(r"[()]|[^\s()]+", parse_tree)][1:])
words = []
constituents = []
index = 0
tree = construct_tree()  # Modifies `words` inplace
construct_constituents(tree)  # Modifies `constituents` inplace
original = reconstruct_original()
assert re.sub(r"\s", "", original)==re.sub(r"\s", "", parse_tree), "Given string and reconstructed are not equal"#
