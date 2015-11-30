import numpy as np
from sklearn import linear_model, cross_validation
class Model:
    def fit(self,X,y):
        self.model.fit(X,y)

    def predict(self,X):
        return self.model.predict(X)

    def cross_val_score(self, X, y):
        return cross_validation.cross_val_score(self.model, X, y, cv=10, scoring='mean_squared_error')

class Linear(Model):
    def __init__(self):
        self.model = linear_model.LinearRegression()

class SGD(Model):
    def __init__(self):
        self.model = linear_model.SGDRegressor()
