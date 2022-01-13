import re

class Tree:

    def __init__(self, name):
        self.children = []
        self.name = name
        # self.start = 0
        # self.end = 0

        def __iter__(self):
            self.n = 0
            return self

        def __next__(self):
            result = self.children[self.n]
            self.n += 1
            return result

def construct_tree():
    while True:
        token = tokens.pop(0)[0]
        if token == ")":
            return node
        if token == "(":
            child = construct_tree()
            node.children.append(child)
        if token in tags:
            node = Tree(token)
        else:
            words.append(token)

def chain_rule(tree, level):

    increment = 1

    if len(tree.children) == 1:
        child = tree.children[0]
        is_not_terminal = len(child.children) != 0
        if is_not_terminal:
            tree = Tree(f"{tree.name}={child.name}")
            tree.children = child.children
            increment = 0

    elif len(tree.children) > 1:
        for i, child in enumerate(tree.children):
                tree.children[i], level = chain_rule(child, level+1)

    return tree, level + increment

def chain(current: Tree) -> Tree:

    if len(current.children) == 1:
        child = current.children[0]
        child_has_children = child.children != []
        if child_has_children:
            tree = Tree(f"{tree.name}={child.name}")
            tree.children = child.children
    return tree

tags = ["DOT", "NNP", "NP", "VBZ", "VP", "S", "TOP"]
parse_tree = "(TOP(S(NP(NNP Ms.)(NNP Haag))(VP(VBZ plays)(NP(NNP Elianti)))(DOT .)))"
tokens = [x for x in re.finditer(r"[()]|[^\s()]+", parse_tree)][1:]
words = []
tree = construct_tree()
tree, level = chain(tree, 0)
tree, level = chain_rule(tree, level)
s = 0










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