import re

class Tree:

    def __init__(self, name):
        self.children = []
        self.name = name
        self.start = 0
        self.end = 0

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        result = self.children[self.n]
        self.n += 1
        return result

    def get_indices(self):
        return [self.start, self.end]

index = 0

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
                node_name = node.name
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


tags = ["DOT", "NNP", "NP", "VBZ", "VP", "S", "TOP"]
parse_tree = "(TOP(S(NP(NNP Ms.)(NNP Haag))(VP(VBZ plays)(NP(NNP Elianti)))(DOT .)))"
parse_tree = "(TOP(S(NP(NNP Ms.)(NNP Haag))(VP(VBZ plays))))"
tokens = [x for x in re.finditer(r"[()]|[^\s()]+", parse_tree)][1:]
words = []
tree = construct_tree()
s = 0



def generate_tagset():
    tags = set()
    path = "/home/marcelbraasch/Downloads/penntreebank/train.txt"
    with open(path, "r") as f:
        func = lambda tree: [x for x in re.finditer(r"[()]|[^\s()]+", tree)]
        for line in f:
            tokens = [x[0] for x in func(line)]
            for i in range(1, len(tokens)):
                if tokens[i-1] == "(":
                    tags.add(tokens[i])
    return tags

#
# constituents = []
#
# def recursion(node: Tree, index: int):
#     curr_children = node.children
#     if len(curr_children) == 1:
#         childs_children = curr_children[0].children
#         if len(childs_children)==0:
#             consituent = (node.node, index, index+1)
#             constituents.append(consituent)
#         elif len(childs_children)>0:
#             if len(curr_children)>0:
#                 merged_node = Tree(f"{node.node}={curr_children[0].node}", curr_children[0])
#                 recursion(merged_node, index)
#             else:
#                 recursion(childs_children, index)
#     else:
#         consituent = (node.node, index, index+len(curr_children)+1)
#         constituents.append(consituent)#
#         for i,child in enumerate(curr_children):
#             recursion(child,index+i)
#
# #recursion(tree,0)
#
#
# data = []
# def recurse(node):
#     if node.children == []:
#         data.append(node.node)
#     else:
#         for child in node.children:
#             recurse(child)
#         data.append(node.node)
#
# recurse(tree)
# s = 0



# # P1
# child1 = Tree("Ms.", [])
# child2 = Tree("Haag", [])
# nnp1 = Tree("NNP", [child1])
# nnp2 = Tree("NNP", [child2])
# np1 = Tree("NP", [nnp1, nnp2])
#
# # P2
# child3 = Tree("plays", [])
# child4 = Tree("Elianti", [])
# vbz1 = Tree("VBZ", [child3])
# nnp3 = Tree("NNP", [child4])
# np2 = Tree("NP", [nnp3])
# vp1 = Tree("VP", [vbz1, np2])
#
# # P3
# child5 = Tree(".", [])
# punkt1 = Tree(".", [child5])
#
# # Sentence level
# S = Tree("S", [np1, vp1, punkt1])
#
# # Root
# tree = Tree("TOP", [S])