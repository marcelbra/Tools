"""
Utilities
"""

from Data import Data
from torch.utils.data import DataLoader
import pickle
import torch
import os
import argparse

# def get_targets2(words, constituents, data, device, x):
#     label_vector = [0] * sum(list(range(len(words) + 1)))
#     for label, start, end in constituents:
#         index = sum([len(words)-i for i in range(end-start-1)])
#         label_vector[index] = data.labelID(label)
#     return torch.Tensor(label_vector).to(torch.int64).to(device)

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

def save_errors(path_errors, epoch, train_wrong, valid_wrong):
    with open(path_errors, "a") as f:
        current = f"Epoch {epoch}:\t" \
                  f"Train errors: {train_wrong}\t" \
                  f"Val errors: {valid_wrong}\n"
        f.write(current)

def save_configs(path_configs, config):
    with open(path_configs, "w") as f:
        f.write(str(config))

def create_directories():
    path_output = "./Outputs/"
    if not os.path.exists(path_output):
        os.mkdir(path_output)
        path_output += "Run-0/"
    else:
        runs = os.listdir(path_output)
        highest = max([int(x.split("-")[-1]) for x in runs])
        path_output += f"Run-{highest + 1}/"
    os.mkdir(path_output)
    return path_output

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dim", type=int)
    parser.add_argument("--word_encoder_hidden_dim", type=int)
    parser.add_argument("--span_encoder_hidden_dim", type=int)
    parser.add_argument("--fc_hidden_dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--span_encoder_num_layers", type=int)
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr_decay", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--validation_amount", type=int)
    parser.add_argument("--decay_from_step", type=int)
    args = parser.parse_args()
    return args
