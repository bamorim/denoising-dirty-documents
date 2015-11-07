from os import walk
from PIL import Image
import numpy as np

def find_files(basedir):
    return sorted([dirname+"/"+filename for (dirname,_,filenames) in walk(basedir) for filename in filenames])

def extract_y(files):
    total = total_sample_count(files)
    y = np.empty((total,1), dtype=np.uint8)

    i = 0
    for f in files:
        for p in Image.open(f).getdata():
            y[i] = p
            i+=1

    return y

def extract_x(files):
    total = total_sample_count(files)
    X = np.empty((total,1), dtype=np.uint8)

    i = 0
    for f in files:
        for p in Image.open(f).getdata():
            X[i] = p
            i+=1

    return X

def total_sample_count(files):
    sample_counts = map(sample_count,files)
    return sum(sample_counts)

def sample_count(f):
    w,h = Image.open(f).size
    return w*h
