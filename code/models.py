import numpy as np
from sklearn import linear_model, cross_validation, tree, neural_network, cluster, metrics
class Model:
    def cross_validation_cv(self):
        return 10

    def fit(self,X,y):
        self.model.fit(X,y)

    def predict(self,X):
        return self.model.predict(X)

    def cross_val_score(self, X, y):
        return cross_validation.cross_val_score(self.model, X, y, cv=self.cross_validation_cv(), scoring='mean_squared_error')

class Linear(Model):
    def __init__(self):
        self.model = linear_model.LinearRegression()

# K-means thresholding
class KMeansThresholding:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters

    def predict(self, X):
        kmeans = cluster.KMeans(n_clusters=self.n_clusters)
        kmeans.fit(X)
        def center_to_val(idx):
            return kmeans.cluster_centers_[idx,0]

        predicted = np.vectorize(center_to_val)(kmeans.predict(X))
        minval = np.min(predicted)
        maxval = np.max(predicted)
        def normalize(X):
            return (X-minval)/(maxval-minval)

        return normalize(predicted)

# ANN

class ANNBasic(Model):
    def cross_validation_cv(self):
        return 3

    def set_model(self,hidden_layer_sizes):
        self.model = neural_network.MLPRegressor(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=hidden_layer_sizes, random_state=1)

class ANN_3(ANNBasic):
    def __init__(self):
        self.set_model((3,))

class ANN_5(ANNBasic):
    def __init__(self):
        self.set_model((5,))

class ANN_3_3(ANNBasic):
    def __init__(self):
        self.set_model((3,3,))

class ANN_9(ANNBasic):
    def __init__(self):
        self.set_model((9,))

class ANN_5_3(ANNBasic):
    def __init__(self):
        self.set_model((5,3,))

class ANN_5_3_5(ANNBasic):
    def __init__(self):
        self.set_model((5,3,5))
