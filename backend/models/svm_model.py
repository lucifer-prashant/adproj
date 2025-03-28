import numpy as np

class SVMModel:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.description = "Support Vector Machine (RBF Kernel)"

    def _rbf_kernel(self, x1, x2, gamma=0.1):
        return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0
        self.X = X
        self.y = y_

        # Gradient descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
        return self

    def predict(self, X):
        output = np.dot(X, self.w) - self.b
        return np.where(output <= 0, 0, 1)

    def predict_proba(self, X):
        distances = np.dot(X, self.w) - self.b
        probs = 1 / (1 + np.exp(-distances))
        return np.column_stack((1 - probs, probs))

    def score(self, X, y):
        return np.mean(self.predict(X) == y)