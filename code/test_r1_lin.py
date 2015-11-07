from sklearn import linear_model
from PIL import Image
from shared import find_files, extract_x_around, filter_point_value
import numpy as np
import pickle

def main():
    test_files = find_files("data/test")
    model = pickle.load(open("data/r1_lin.p", "rb"))
    for f in test_files:
        w,h = Image.open(f).size
        X = extract_x_around([f],1)
        y = model.predict(X)
        y = np.reshape(y, (h-2,w-2))
        y = np.vectorize(filter_point_value)(y)

        img = Image.fromarray(y.astype(np.uint8))
        outf = str.replace(f,"data/test","out/r1_lin")
        img.save(outf)

main()
