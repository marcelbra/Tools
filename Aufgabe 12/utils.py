from Data import Data
import pickle

def load_config(model_config_path):
    with open(model_config_path, "r") as f:
        return eval(f.readlines()[0])

def get_sentences_from_test(path):
    with open(path, "r") as f:
        return [[x[-1] for x in list(map(lambda x: x.split(), line.split(")"))) if x]
                for line in f]

def load_from_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_data(path):
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except:
        pass
        # data = Data(paths["path_train"], paths["path_dev"])
        # with open(paths["path_data"], "wb") as handle:
        #      pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # return data

def build_labels_vector(n):
    label_vector = []
    for i in range(1, n + 1):
        label_vector.extend([i] * (n- i + 1))
    return label_vector