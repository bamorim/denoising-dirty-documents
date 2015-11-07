from sklearn import linear_model
from PIL import Image
from shared import find_files, extract_x
import numpy as np
import pickle

def main():
    test_files = find_files("data/test")
    model = pickle.load(open("data/linear_model.p", "rb"))
    for f in test_files:
        w,h = Image.open(f).size
        X = extract_x([f])
        y = np.reshape(model.predict(X), (h,w))
        y = np.vectorize(filter_point_value)(y)

        img = Image.fromarray(y.astype(np.uint8))
        outf = str.replace(f,"data/test","out/linear")
        img.save(outf)

def filter_point_value(val):
    if(val < 0):
        return 0

    if(val > 255):
        return 255

    return int(val)

main()
