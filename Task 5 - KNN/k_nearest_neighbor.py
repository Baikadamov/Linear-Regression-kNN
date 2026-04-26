import numpy as np
import math

class KNearestNeighbor:
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def compute_distances_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i, j] =  math.sqrt(sum((X[i] - self.X_train[j])**2))
        return dists
    
    def compute_distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i, :] = np.sqrt(np.sum((X[i] - self.X_train) ** 2, axis=1))
        return dists

    def compute_distances_no_loops(self, X):
        test_sq = np.sum(X ** 2, axis=1, keepdims=True)        # (num_test, 1)
        train_sq = np.sum(self.X_train ** 2, axis=1)            # (num_train,)
        cross = X @ self.X_train.T                              # (num_test, num_train)
        dists = np.sqrt(test_sq - 2 * cross + train_sq)
        return dists

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = self.y_train[np.argsort(dists[i])[:k]]
            y_pred[i] = np.bincount(closest_y.astype(int)).argmax()
        return y_pred

    def predict(self, X, k=1):
        dists = self.compute_distances_no_loops(X)
        return self.predict_labels(dists, k=k)

