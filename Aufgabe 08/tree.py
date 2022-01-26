class Tree:

    def __init__(self, name):
        self.children = []
        self.name = name
        self.start = 0
        self.end = 0

    def get_indices(self):
        return [self.start, self.end]
