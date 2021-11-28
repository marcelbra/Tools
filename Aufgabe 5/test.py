from model import LogLinear
import sys

if __name__ == "__main__":
    assert len(sys.argv) == 3, "Call script as $ python3 test.py paramfile test-dir"
    ll = LogLinear(mode="test", data_dir=sys.argv[2], paramfile=sys.argv[1])
    ll.load_parameters()
    ll.predict()
