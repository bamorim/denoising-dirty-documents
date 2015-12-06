import numpy as np
import sys
from models import *
from shared import getarg

def main():
    inname = getarg(1,"data/train_raw.npz")
    model_name = getarg(2, "Linear")
    model = eval(model_name)()

    print("Train Set: "+inname)
        
    print("Loading file")
    files = np.load(inname)
    X = files['X']
    y = files['y']

    print("Validating...")
    scores = model.cross_val_score(X,y)
    print("Model: "+model_name)
    print("MSE: %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std() * 200))

main()
