from math import sqrt
import numpy as np
from collections import Counter


def calc_equlidean_dist( row1, row2 ):
    dist = sqrt(np.sum((row1 - row2) ** 2))
    return dist


class KNN_classifier:
    def __init__( self, k ):
        self.k = k

    def fit( self, X, y ):
        self.X_train = X
        self.y_train = y

    def predict( self, X ):
        y_predicted = [self.do_prediction(x) for x in X]
        return np.array(y_predicted)

    def do_prediction( self, x ):
        distance = [calc_equlidean_dist(x, x_train) for x_train in self.X_train]
        idx_k = np.argsort(distance)[:self.k]  # calculating indices of k nearest samples (limited only for (0,k) range
        labels_k = [self.y_train[idx] for idx in idx_k]

        most_common_labels = Counter(labels_k).most_common(1)
        return most_common_labels[0][0]
