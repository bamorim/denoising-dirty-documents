import numpy as np
from sklearn import metrics
from shared import extract_x, extract_y, find_files
from models import KMeansThresholding
def main():
    model = KMeansThresholding()
    
    train_files_x = find_files("data/train")
    train_files_y = find_files("data/train_cleaned")

    y_pred = map(lambda x: extract_x([x],0), train_files_x)
    y_pred = list(map(predict, y_pred))
    y_pred = np.concatenate(y_pred)

    y_true = np.concatenate(list(map(lambda x: extract_y([x],0) ,train_files_y)))

    print("Calculating MSE")

    mse =  metrics.mean_squared_error(y_true, y_pred)
    print("MSE: %0.2f (+/- %0.2f)" % (mse.mean()*100, mse.std() * 200))

def predict(X):
    model = KMeansThresholding(3)
    return model.predict(X)

main()
