import numpy as np
from collections import Counter

class NaiveBayesModel:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None
        self.description = "Improved Gaussian Naïve Bayes"

    def fit(self, X, y):
        """Train the Gaussian Naïve Bayes model."""
        self.classes = np.unique(y)  # Unique class labels
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        for i, c in enumerate(self.classes):
            X_c = X[y == c]  # Subset of X for class `c`
            self.mean[i, :] = X_c.mean(axis=0)
            self.var[i, :] = X_c.var(axis=0) + 1e-3  # 🔥 Increased smoothing
            self.priors[i] = X_c.shape[0] / X.shape[0]  # Class prior

        return self

    def _calculate_log_likelihood(self, X):
        """Calculate log probability for each class."""
        log_likelihoods = np.zeros((X.shape[0], len(self.classes)))

        for i, c in enumerate(self.classes):
            log_variance = np.log(self.var[i, :])
            log_prior = np.log(self.priors[i])

            # 🔥 Compute log likelihood using Gaussian distribution
            log_likelihoods[:, i] = -0.5 * np.sum(
                log_variance + ((X - self.mean[i, :]) ** 2) / self.var[i, :], axis=1
            )
            log_likelihoods[:, i] += log_prior  # Add log prior

        return log_likelihoods

    def predict(self, X):
        """Predict class labels."""
        log_likelihoods = self._calculate_log_likelihood(X)
        return self.classes[np.argmax(log_likelihoods, axis=1)]

    def predict_proba(self, X):
        """Predict class probabilities (softmax applied)."""
        log_likelihoods = self._calculate_log_likelihood(X)
        exp_likelihoods = np.exp(log_likelihoods - np.max(log_likelihoods, axis=1, keepdims=True))  # Avoid overflow
        return exp_likelihoods / np.sum(exp_likelihoods, axis=1, keepdims=True)  # Normalize

    def score(self, X, y):
        """Compute accuracy of the model."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
