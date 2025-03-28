import numpy as np
from collections import Counter

class DecisionTreeBase:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def _best_split(self, X, y):
        m = X.shape[0]
        if m <= 1:
            return None, None

        num_parent = [np.sum(y == c) for c in np.unique(y)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None

        for idx in range(X.shape[1]):
            thresholds = np.unique(X[:, idx])
            for threshold in thresholds:
                left_y = y[X[:, idx] <= threshold]
                right_y = y[X[:, idx] > threshold]
                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                left_counts = [np.sum(left_y == c) for c in np.unique(y)]
                right_counts = [np.sum(right_y == c) for c in np.unique(y)]
                gini_left = 1.0 - sum((n / len(left_y)) ** 2 for n in left_counts)
                gini_right = 1.0 - sum((n / len(right_y)) ** 2 for n in right_counts)
                gini = (len(left_y) * gini_left + len(right_y) * gini_right) / m

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = threshold

        return best_idx, best_thr

    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)
        n_labels = len(np.unique(y))

        if (self.max_depth is not None and depth >= self.max_depth) or n_labels == 1:
            leaf_value = Counter(y).most_common(1)[0][0]
            return self.Node(value=leaf_value)

        feature_idx, threshold = self._best_split(X, y)

        if feature_idx is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return self.Node(value=leaf_value)

        left_idxs = X[:, feature_idx] <= threshold
        right_idxs = X[:, feature_idx] > threshold
        left = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)

        return self.Node(feature_idx, threshold, left, right)

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class RandomForestModel:
    def __init__(self, n_trees=10, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        self.description = "Random Forest Ensemble"

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeBase(max_depth=self.max_depth)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.tree = tree._build_tree(X_sample, y_sample)
            self.trees.append(tree)
        return self

    def predict(self, X):
        predictions = np.array([self._predict_single(x) for x in X])
        return predictions

    def _predict_single(self, x):
        predictions = [tree._traverse_tree(x, tree.tree) for tree in self.trees]
        return Counter(predictions).most_common(1)[0][0]

    def predict_proba(self, X):
        probas = []
        for x in X:
            predictions = [tree._traverse_tree(x, tree.tree) for tree in self.trees]
            counts = Counter(predictions)
            proba = np.zeros(len(self.classes))
            for i, cls in enumerate(self.classes):
                proba[i] = counts[cls] / self.n_trees
            probas.append(proba)
        return np.array(probas)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)