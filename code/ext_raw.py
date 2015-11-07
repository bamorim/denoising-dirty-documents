from PIL import Image
import numpy as np
from shared import find_files, extract_x, extract_y

def main():
    train_files_x = find_files("data/train")
    X = extract_x(train_files_x)

    train_files_y = find_files("data/train_cleaned")
    y = extract_y(train_files_y)

    np.savez("data/train_raw.npz",X=X,y=y)

def train_model(train_files_x, train_files_y):
    model = linear_model.LinearRegression()
    X = extract_x(train_files_x)
    y = extract_y(train_files_y)
    model.fit(X,y)
    return lambda V: model.predict(V)



main()
