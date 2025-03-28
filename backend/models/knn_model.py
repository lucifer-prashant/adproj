import numpy as np
from collections import Counter

class KNNModel:
    def __init__(self, n_neighbors=5):
        self.k = n_neighbors
        self.description = "K-Nearest Neighbors"
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict_single(self, x):
        # Calculate distances between x and all examples in the training set
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Sort by distance and return indices of first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Extract the labels of k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def predict_proba(self, X):
        probas = []
        classes = np.unique(self.y_train)
        
        for x in X:
            distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            # Calculate probability for each class
            class_counts = Counter(k_nearest_labels)
            proba = np.zeros(len(classes))
            for i, cls in enumerate(classes):
                proba[i] = class_counts[cls] / self.k
            probas.append(proba)
            
        return np.array(probas)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)