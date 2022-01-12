"""


"""


class Tree:

    def __init__(self, node, children):

        self.node = node
        self.children = children

    def __len__(self):
        return len(self.children)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        result = self.children[self.n]
        self.n += 1
        return result

# P1
child1 = Tree("Ms.", [])
child2 = Tree("Haag", [])
nnp1 = Tree("NNP", [child1])
nnp2 = Tree("NNP", [child2])
np1 = Tree("NP", [nnp1, nnp2])

# P2
child3 = Tree("plays", [])
child4 = Tree("Elianti", [])
vbz1 = Tree("VBZ", [child3])
nnp3 = Tree("NNP", [child4])
np2 = Tree("NP", [nnp3])
vp1 = Tree("VP", [vbz1, np2])

# P3
child5 = Tree(".", [])
punkt1 = Tree(".", [child5])

# Sentence level
S = Tree("S", [np1, vp1, punkt1])

# Root
tree = Tree("TOP", [S])

constituents = []

def recursion(node: Tree, index: int):
    curr_children = node.children
    if len(curr_children) == 1:
        childs_children = curr_children[0].children
        if len(childs_children)==0:
            consituent = (node.node, index, index+1)
            constituents.append(consituent)
        elif len(childs_children)>0:
            if len(curr_children)>0:
                merged_node = Tree(f"{node.node}={curr_children[0].node}", curr_children[0])
                recursion(merged_node, index)
            else:
                recursion(childs_children, index)
    else:
        consituent = (node.node, index, index+len(curr_children)+1)
        constituents.append(consituent)#
        for i,child in enumerate(curr_children):
            recursion(child,index+i)

#recursion(tree,0)


data = []
def recurse(node):
    if node.children == []:
        data.append(node.node)
    else:
        for child in node.children:
            recurse(child)
        data.append(node.node)

recurse(tree)
s = 0