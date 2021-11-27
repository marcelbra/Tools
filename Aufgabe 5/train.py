from model import LogLinear
import sys

if __name__ == "__main__":
    assert len(sys.argv) == 3, "Call script as $ python3 test.py paramfile mail-dir"
    ll = LogLinear(mode="train", paramfile=sys.argv[1], data_dir=sys.argv[2])
    ll.fit()
    #ll.save_parameters()