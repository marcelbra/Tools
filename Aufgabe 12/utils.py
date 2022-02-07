from Data import Data
import pickle

def load_config(model_config_path):
    with open(model_config_path, "r") as f:
        return eval(f.readlines()[0])

def get_test_data(path):
    sentences, trees = [], []
    with open(path, "r") as f:
        for line in f:
            sentences.append([x[-1] for x in list(map(lambda x: x.split(), line.split(")"))) if x])
            trees.append(line)
    return sentences, trees

def load_data(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except:
        return Data(paths["path_train"], paths["path_dev"])

class Index:

    def __init__(self, amount):
        """Builds a 1D index for a specific sentence length given
        the start and end (so 2D) of a consitutuent. This is needed
        because model logits are returned in 1D and we need to convert."""
        self.labels_vector = self.build_labels_vector(amount)

    def __call__(self, *args, **kwargs):
        start, end = args
        return self.labels_vector.index(end - start) + start

    @staticmethod
    def build_labels_vector(n):
        label_vector = []
        for i in range(1, n + 1):
            label_vector.extend([i] * (n - i + 1))
        return label_vector

