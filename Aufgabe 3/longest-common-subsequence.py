"""
P7 Experimente, Evaluierung und Tools
Aufgabe 3 - Longest Common Subsequence

Gruppe:
Marcel Braasch
Nadja Seeberg
Sinem Kühlewind (geb. Demiraslan)
"""

import sys


def lcs_table(s1, s2):
    """
    Expects two strings s1 and s2.
    Initializes a len(s1)+1 x len(s2)+1 table to save how many common
    characters in the previous substring exist. Table is initalized with
    extra 0s in the beginning which denote the empty string. For each entry
    check whether characters coincide, look for the max value in the
    surrounding fields (above, to the left, diagonally to the left)
    in terms of characters coinciding. If characters are not
    equal just propagate the previous max value without increasing.Otherwise, increase by 1.
    """
    length_1, length_2 = len(s1) + 1, len(s2) + 1
    table = [[0] * length_1 for _ in range(length_2)]
    for i in range(1, length_1):
        for j in range(1, length_2):
            if s1[i - 1] == s2[j - 1]:
                table[j][i] = 1 + max(table[j - 1][i], table[j][i - 1], table[j - 1][i - 1])
            else:
                table[j][i] = max(table[j - 1][i], table[j][i - 1], table[j - 1][i - 1])
    return table


def backtrack(s1, s2, table):
    """
    Backtrack through a filled LCS table. Starts in the bottom right corner
    and checks whether we can make a diagonal move. If so, the character is
    part of LCS, if not check if we can go up, if so, do it. If not go left.
    Repeated until the end of the table. Lastly, return the reversed sequence.
    """

    i, j = len(s1), len(s2)
    sequence = ""
    while True:
        if table[j][i - 1] == table[j - 1][i] == table[j - 1][i - 1]:
            sequence += s1[i - 1]
            i, j = i - 1, j - 1
        elif table[j - 1][i] == table[j][i]:
            j -= 1
        else:
            i -= 1
        if not i or not j:
            return sequence[::-1]


def main(s1, s2):
    table = lcs_table(s1, s2)
    lcs = backtrack(s1, s2, table)
    print(lcs)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        raise RuntimeError("script should be invoked with two arguments.")
