"""
Defines simple vector operations on lists.
"""


def add(list_1, list_2):
    return [x + list_2[i] for i, x in enumerate(list_1)]

def sub(list_1, list_2):
    return [x - list_2[i] for i, x in enumerate(list_1)]

def mul(list_1, list_2):
    return [x * list_2[i] for i, x in enumerate(list_1)]

def div(list_1, list_2):
    return [x / list_2[i] for i, x in enumerate(list_1)]

def dot(list_1, list_2):
    return sum(mul(list_1, list_2))

def create_vec(value, n):
    return [value] * n