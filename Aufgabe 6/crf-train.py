from model import CRFTagger
import sys

def train():
    train_file = sys.argv[1]
    param_file = sys.argv[2]
    crf = CRFTagger(train_file, param_file)
    crf.fit()

train()