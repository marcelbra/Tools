from model import CRFTagger
import sys

def test():
    data_file = sys.argv[2]
    param_file = sys.argv[1]
    crf = CRFTagger(data_file, param_file)
    crf.predict()

test()