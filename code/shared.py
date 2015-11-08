from os import walk
from PIL import Image
import numpy as np

def find_files(basedir):
    return sorted([dirname+"/"+filename for (dirname,_,filenames) in walk(basedir) for filename in filenames])

def extract_y(files,radius=0):
    total = total_sample_count(files,radius)
    y = np.empty((total,1), dtype=np.uint8)

    sample_i = 0
    for f in files:
        image = Image.open(f)
        w,h = image.size
        pixels = np.reshape(np.array(image.getdata()),(h,w));
        for i in range(radius,h-radius):
            for j in range(radius,w-radius):
                y[sample_i]=pixels[i,j]
                sample_i += 1

    return y

def extract_x(files,radius=0):
    total = total_sample_count(files,radius)
    X = np.empty((total,(1+2*radius)**2), dtype=np.uint8)

    sample_i = 0
    for f in files:
        image = Image.open(f)
        w,h = image.size
        pixels = np.reshape(np.array(image.getdata()),(h,w));
        for i in range(radius,h-radius):
            for j in range(radius,w-radius):
                sample_j = 0
                for ri in range(-radius,radius):
                    for rj in range(-radius,radius):
                        X[sample_i,sample_j]=pixels[i+ri,j+rj]
                        sample_j += 1
                sample_i += 1

    return X
    

def total_sample_count(files,radius=0):
    sample_counts = map(lambda f: sample_count(f,radius),files)
    return sum(sample_counts)

def sample_count(f,radius=0):
    w,h = Image.open(f).size
    return (w-2*radius)*(h-2*radius)

def filter_point_value(val):
    if(val < 0):
        return 0

    if(val > 255):
        return 255

    return int(val)
