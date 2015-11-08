import numpy as np
import sys
from shared import find_files, extract_x, extract_y

def main():
    try:
        r = int(sys.argv[1])
    except Exception:
        r = 0

    print("Extracting with radius=%d" % (r))

    train_files_x = find_files("data/train")
    X = extract_x(train_files_x,r)

    train_files_y = find_files("data/train_cleaned")
    y = extract_y(train_files_y,r)

    outname = "data/train_r"+str(r)+".npz"
    np.savez(outname,X=X,y=y)

def train_model(train_files_x, train_files_y):
    model = linear_model.LinearRegression()
    X = extract_x(train_files_x)
    y = extract_y(train_files_y)
    model.fit(X,y)
    return lambda V: model.predict(V)



main()
