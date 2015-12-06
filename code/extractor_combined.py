import numpy as np
from shared import find_files, extract_x, extract_y, getarg

def main():
    r = int(getarg(1, 0))
    k = int(getarg(2, 3))

    print("Extracting with radius=%d" % (r))

    prefix = "data/train"
    outname = prefix+"_c_k"+str(k)+"_r"+str(r)+".npz"
    print("Saving to: "+ outname)

    train_files_x = find_files(prefix)
    X = extract_x(train_files_x,r)
    Xk = extract_x(find_files(prefix+"_k"+str(k)),r,True)
    X = np.concatenate([X,Xk], axis=1)

    train_files_y = find_files("data/train_cleaned")
    y = extract_y(train_files_y,r)

    np.savez(outname,X=X,y=y)

main()
