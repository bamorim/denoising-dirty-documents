import numpy as np
import sys
from models import *

def main():
    if len(sys.argv) > 1:
        inname = sys.argv[1]
    else:
        inname = "data/train_raw.npz"

    if len(sys.argv) > 2:
        model_name = sys.argv[2]
    else:
        model_name = "Linear"

        
    files = np.load(inname)
    X = files['X']
    y = files['y']
    model = eval(model_name)()
    scores = model.cross_val_score(X,y)
    print("Train Set: "+inname)
    print("Model: "+model_name)
    print("RMS: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

main()
