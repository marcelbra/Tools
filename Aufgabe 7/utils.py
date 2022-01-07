"""

"""


import pandas as pd
from collections import Counter, defaultdict

def get_words_map_and_max_seq_len(dirs, k=5000):
    max_seq_len = float("-inf")
    words = Counter()
    for name, path in dirs.items():
        data = pd.read_csv(path, sep="\t", header=None)[1].tolist()  # Get list of sentences
        data = [x.split() for x in data]  # Split into words
        curr_seq_len = max([len(x) for x in data])  # Get and update current max
        if curr_seq_len > max_seq_len:
            max_seq_len = curr_seq_len
        data = [item for sublist in data for item in sublist]  # Flatten
        words += Counter(data)
    # Start at 2 because 0 will be for padding and 1 for unknown
    most_common = {x:i for i, (x,_) in enumerate(words.most_common(k), 2)}
    most_common = defaultdict(lambda:1, most_common)
    return most_common, max_seq_len
