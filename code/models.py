import numpy as np
from sklearn import linear_model
class Model:
    def fit(self,X,y):
        self.model.fit(X,y)

    def predict(self,X):
        return self.model.predict(X)

class Linear(Model):
    def __init__(self):
        self.model = linear_model.LinearRegression()

class SGD(Model):
    def __init__(self):
        self.model = linear_model.SGDRegressor(eta0=0.0001)
