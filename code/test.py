from models import *
from PIL import Image
from shared import find_files, extract_x, denormalize, getarg
from os import path
import sys
import numpy as np
import pickle

def main():
    f = sys.argv[1]
    r = int(getarg(2,0))
    inname = getarg(3,"data/r%d_lin.p" % (r))

    print("Test with radius=%d and file %s and model %s" % (r,f,inname))

    model = pickle.load(open(inname, "rb"))

    w,h = Image.open(f).size
    X = extract_x([f],r)

    dim = (h-2*r,w-2*r)
    y = model.predict(X)
    y = np.reshape(y, dim)
    y = denormalize(y)

    if(np.any(np.isnan(y))):
        print("result of",f, "has nan values after denormalizing")

    img = Image.fromarray(y.astype(np.uint8))
    outfold = str.replace(str.replace(inname,".p",""),"data/","out/")
    outf = path.join(outfold,path.basename(f))
    img.save(outf)

main()
