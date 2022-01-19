from model import NaiveBayes
import sys

if __name__ == "__main__":
    assert len(sys.argv) == 3, "Call script as python3 train.py train-dir paramfile"
    nb = NaiveBayes(mode="test")
    nb.fit()
    nb.predict()