import numpy as np
from sklearn import linear_model
import pickle

def main():
    files = np.load("data/train_raw.npz")
    X = files['X']
    X2 = np.vectorize(lambda x: x**2)(X)
    X = np.concatenate([X,X2], axis=1)
    y = files['y']
    model = linear_model.LinearRegression()
    model.fit(X,y)
    pickle.dump(model, open("data/raw_sq.p", "wb"))

main()
