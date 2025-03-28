import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, class_counts=None):
        self.feature = feature      # Index of feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.left = left          # Left child node
        self.right = right        # Right child node
        self.value = value        # Predicted class (for leaf nodes)
        self.class_counts = class_counts  # Class distributions for probability

class DecisionTreeModel:
    def __init__(self, max_depth=10):
        self.root = None
        self.max_depth = max_depth
        self.description = "Manual Decision Tree"
        self.classes_ = None  # Stores unique classes seen in training

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _information_gain(self, parent, left_child, right_child):
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        return self._entropy(parent) - (weight_left * self._entropy(left_child) + weight_right * self._entropy(right_child))

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) > 0 and np.sum(right_mask) > 0:
                    gain = self._information_gain(y, y[left_mask], y[right_mask])
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or n_classes == 1 or n_samples < 2):
            class_counts = Counter(y)
            total_samples = len(y)
            probabilities = {cls: count / total_samples for cls, count in class_counts.items()}
            leaf_value = max(class_counts, key=class_counts.get)
            return Node(value=leaf_value, class_counts=probabilities)

        # Find best split
        best_feature, best_threshold = self._best_split(X, y)

        if best_feature is None:
            class_counts = Counter(y)
            total_samples = len(y)
            probabilities = {cls: count / total_samples for cls, count in class_counts.items()}
            leaf_value = max(class_counts, key=class_counts.get)
            return Node(value=leaf_value, class_counts=probabilities)

        # Create child nodes
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def fit(self, X, y):
        self.classes_ = np.unique(y)  # Store class labels
        self.root = self._build_tree(X, y)
        return self

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root).value for x in X])

    def predict_proba(self, X):
        predictions = []
        for x in X:
            leaf = self._traverse_tree(x, self.root)
            prob_dist = np.zeros(len(self.classes_))  # Ensure correct shape
            for cls, prob in leaf.class_counts.items():
                prob_dist[np.where(self.classes_ == cls)[0][0]] = prob
            predictions.append(prob_dist)
        return np.array(predictions)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)  # Classification accuracy
