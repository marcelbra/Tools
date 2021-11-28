from model import LogLinear
import sys

if __name__ == "__main__":
    assert len(sys.argv) == 3, "Call script as $ python3 train.py train-dir paramfile"
    ll = LogLinear(mode="train", data_dir=sys.argv[1], paramfile=sys.argv[2])
    ll.fit()
    ll.save_parameters()