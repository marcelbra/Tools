from model import NaiveBayes
import sys

if __name__ == "__main__":
    assert len(sys.argv) == 3, "Call script as $ python3 test.py paramfile mail-dir"
    nb = NaiveBayes(mode="train")
    nb.fit()
    nb.save_parameters()