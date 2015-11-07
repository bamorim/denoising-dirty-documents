from sklearn import linear_model
from PIL import Image
from shared import find_files, extract_x, filter_point_value
import numpy as np
import pickle

def main():
    test_files = find_files("data/test")
    model = pickle.load(open("data/raw_sq.p", "rb"))
    for f in test_files:
        w,h = Image.open(f).size
        X = extract_x([f])
        X2 = np.vectorize(lambda x: x**2)(X)
        X = np.concatenate([X,X2], axis=1)
        y = np.reshape(model.predict(X), (h,w))
        y = np.vectorize(filter_point_value)(y)

        img = Image.fromarray(y.astype(np.uint8))
        outf = str.replace(f,"data/test","out/raw_sq")
        img.save(outf)

main()
