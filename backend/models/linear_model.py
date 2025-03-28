import numpy as np

class LinearRegressionModel:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.description = "Manual Linear Regression"

    def fit(self, X, y, learning_rate=0.01, n_iterations=1000):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
        
        return self

    def predict(self, X):
        return (np.dot(X, self.weights) + self.bias > 0.5).astype(int)  # Convert to binary output

    def predict_proba(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        prob = 1 / (1 + np.exp(-y_pred))  # Apply sigmoid to get probability
        return np.vstack((1 - prob, prob)).T  # Convert to probability format

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)  # Classification accuracy
