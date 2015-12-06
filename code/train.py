import numpy as np
import pickle
import sys
from models import *
from shared import getarg

def main():
    inname = getarg(1,"data/train_r0.npz")
    outname = getarg(2,"data/r0_lin.p")
    modelname = getarg(3, "Linear")
        
    files = np.load(inname)
    X = files['X']
    y = files['y']
    model = eval(modelname)()
    model.fit(X,y)
    pickle.dump(model, open(outname, "wb"))

main()
