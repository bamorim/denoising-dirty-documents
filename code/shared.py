from os import walk
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import sys

import os, errno

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def getarg(i,default):
    if len(sys.argv) > i:
        return sys.argv[i]
    else:
        return default

def find_files(basedir):
    return sorted([dirname+"/"+filename for (dirname,_,filenames) in walk(basedir) for filename in filenames])

def denormalize_pix(pix):
    if pix < 0:
        return 0
    elif pix > 1:
        return 255
    else:
        return int(pix*255)

def denormalize(X):
    return np.vectorize(denormalize_pix, otypes=[np.uint8])(X)

def extract_y(files,radius=0):
    total = total_sample_count(files,radius)
    y = np.empty((total,1), dtype=np.float64)

    sample_i = 0
    for f in files:
        image = Image.open(f)
        w,h = image.size
        pixels = np.reshape(np.array(image.getdata()),(h,w));
        for i in range(radius,h-radius):
            for j in range(radius,w-radius):
                y[sample_i]=pixels[i,j]/255
                sample_i += 1
    return y

def extract_x(files,radius=0,cut_only=False):
    total = total_sample_count(files,radius)

    if(cut_only):
        X = np.empty((total,1), dtype=np.float64)
    else:
        X = np.empty((total,(1+2*radius)**2), dtype=np.float64)

    sample_i = 0
    for f in files:
        image = Image.open(f)
        w,h = image.size
        pixels = np.reshape(np.array(image.getdata()),(h,w));
        for i in range(radius,h-radius):
            for j in range(radius,w-radius):
                sample_j = 0
                if(cut_only):
                    X[sample_i,0]=pixels[i,j]
                else:
                    for ri in range(-radius,radius+1):
                        for rj in range(-radius,radius+1):
                            X[sample_i,sample_j]=pixels[i+ri,j+rj]/255
                            sample_j += 1
                sample_i += 1
    return X
    

def total_sample_count(files,radius=0):
    sample_counts = map(lambda f: sample_count(f,radius),files)
    return sum(sample_counts)

def sample_count(f,radius=0):
    w,h = Image.open(f).size
    return (w-2*radius)*(h-2*radius)
