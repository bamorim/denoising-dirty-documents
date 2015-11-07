import numpy as np
from sklearn import linear_model
import pickle

def main():
    files = np.load("data/train_linear.npz")
    X = files['X']
    y = files['y']
    model = linear_model.LinearRegression()
    model.fit(X,y)
    pickle.dump(model, open("raw_lin.p", "wb"))

main()
