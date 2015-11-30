import numpy as np
import pickle
import sys
from models import *

def main():
    if len(sys.argv) > 2:
        inname = sys.argv[1]
        outname = sys.argv[2]
    else:
        inname = "data/train_raw.npz"
        outname = "data/raw_lin.p"
        
    files = np.load(inname)
    X = files['X']
    y = files['y']
    model = Linear()
    model.fit(X.astype(np.float32),y.astype(np.float32))
    pickle.dump(model, open(outname, "wb"))

main()
