"""


"""

from Data import Data
import pickle
import torch

def get_targets(words, constituents, data, device):
    # Build labels vector
    label_vector = []
    for i in range(1, len(words) + 1):
        label_vector.extend([i] * (len(words) - i + 1))

    # Save index and label of respective constituent
    consts = []
    labels = []
    for constituent in constituents:
        label, start, end = constituent
        consts.append(label_vector.index(end - start) + start)
        labels.append(label)

    # Add label ID if it is a constituent else 0
    for i in range(len(label_vector)):
        if i in consts:
            label = labels[consts.index(i)]
            label_vector[i] = data.labelID(label)
        else:
            label_vector[i] = 0

    return torch.Tensor(label_vector).to(torch.int64).to(device)


def load_data(paths):
    try:
        with open(paths["path_data"], "rb") as handle:
            return pickle.load(handle)
    except:
        data = Data(paths["path_train"], paths["path_dev"])
        with open(paths["path_data"], "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return data

def save_errors(path_errors, epoch, train_metrics, valid_metrics):
    with open(path_errors, "a") as f:
        current = f"Epoch {epoch}:\t" \
                  f"Train errors: {train_metrics['wrong']}\t" \
                  f"Val errors: {valid_metrics['wrong']}\n"
        f.write(current)