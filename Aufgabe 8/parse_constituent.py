import re

class Tree:

    def __init__(self, name):
        self.children = []
        self.name = name
        self.start = 0
        self.end = 0

    def get_indices(self):
        return [self.start, self.end]

def generate_tagset():
    tags = set()
    path = "/home/marcelbraasch/Downloads/penntreebank/train.txt"
    with open(path, "r") as f:
        func = lambda tree: [x for x in re.finditer(r"[()]|[^\s()]+", tree)]
        for line in f:
            tokens = [x[0] for x in func(line)]
            for i in range(1, len(tokens)):
                if tokens[i - 1] == "(":
                    tags.add(tokens[i])
    return tags

def construct_tree():
    global index
    while True:
        token = tokens.pop(0)[0]
        if token == ")":
            return node
        elif token == "(":
            child = construct_tree()
            has_other_children = node.children == []
            node.children.append(child)
            next_token = tokens[0][0]
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

def construct_constituents(node):
    """Do in-order traversal to construct the constituent list."""
    constituents.append((node.name, node.start, node.end))
    for child in node.children:
        construct_constituents(child)

def construct_original_string():
    closing = []
    tree = "("
    for constituent in constituents:
        name = constituent[0].replace("=", "(")
        start, end = constituent[1], constituent[2]
        if tree[-1] != "(":
            tree += "("
        tree += name
        diff = end - start
        if diff == 1:
            tree += " " + words.pop(0) + ")"
            closing = [x-1 for x in closing]
        else:
            tree += "("
            closing.append(diff)
        tree += ")" * closing.count(0)
        closing = [x for x in closing if x]
    return tree


tags = ["DOT", "NNP", "NP", "VBZ", "VP", "S", "TOP"]
parse_tree = "(TOP(S(NP(NNP Ms.)(NNP Haag))(VP(VBZ plays)(NP(NNP Elianti)))(DOT .)))"
tokens = [x for x in re.finditer(r"[()]|[^\s()]+", parse_tree)][1:]
words = []
constituents = []
index = 0
tree = construct_tree()
construct_constituents(tree)
original = construct_original_string()
#