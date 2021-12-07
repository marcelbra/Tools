"""
Helper file for various mathematical operations.
"""

from typing import List, Union
import math

Numeric = Union[int, float]
Vector = List[Numeric]

def add(list_1: Vector,
        list_2: Vector
        ) -> Numeric:
    return [x + list_2[i] for i, x in enumerate(list_1)]

def sub(list_1: Vector,
        list_2: Vector
        ) -> Numeric:
    return [x - list_2[i] for i, x in enumerate(list_1)]

def mul(list_1: Vector,
        list_2: Vector
        ) -> Numeric:
    return [x * list_2[i] for i, x in enumerate(list_1)]

def div(list_1: Vector,
        list_2: Vector
        )-> Numeric:
    return [x / list_2[i] for i, x in enumerate(list_1)]

def dot(list_1: Vector,
        list_2: Vector
        ) -> Numeric:
    return sum(mul(list_1, list_2))

def create_vec(value: Numeric,
               n: Numeric
               ) -> Vector:
    return [value] * n

def log_sum_exp(a: Numeric,
                b: Numeric
                ) -> Numeric:
    if b > a: a, b = b, a
    return a + math.log(1+math.exp(b-a))