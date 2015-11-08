from models import *
from PIL import Image
from shared import find_files, extract_x, filter_point_value
import sys
import numpy as np
import pickle

def main():
    try:
        r = int(sys.argv[1])
    except Exception:
        r = 0
    print("Test with radius=%d" % (r))

    test_files = find_files("data/test")
    model = pickle.load(open("data/r%d_lin.p" % (r), "rb"))
    for f in test_files:
        w,h = Image.open(f).size
        X = extract_x([f],r)
        dim = (h-2*r,w-2*r)
        print(X.shape)
        print(dim)
        y = model.predict(X)
        print(y.shape)
        y = np.reshape(y, dim)
        y = np.vectorize(filter_point_value)(y)

        img = Image.fromarray(y.astype(np.uint8))
        outf = str.replace(f,"data/test","out/r%d_lin" % (r))
        img.save(outf)

main()
