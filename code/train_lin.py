import numpy as np
from sklearn import linear_model
import pickle
import sys

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
    model = linear_model.LinearRegression()
    model.fit(X,y)
    pickle.dump(model, open(outname, "wb"))

main()
