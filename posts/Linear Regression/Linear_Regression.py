import numpy as np


class Linear_Regression:
    def __init__(self):
        self.coefficients = None

    def analytic_fit(self, X, y):
        X_ = np.hstack((np.ones((X.shape[0], 1)), X))  # padding for bias term
        self.coefficients = np.linalg.inv(X_.T @ X_) @ X_.T @ y

    def gradient_fit(self, X, y, learning_rate=0.0001, max_iterations=10000):
        X_ = np.hstack((np.ones((X.shape[0], 1)), X))  # padding for bias term
        self.coefficients = np.random.rand(X_.shape[1])  # initializing with random weights
        self.score_history = []  

        P = X_.T @ X_
        q = X_.T @ y

        for _ in range(max_iterations):
            gradient = (P @ self.coefficients - q)
            self.coefficients -= learning_rate * gradient

            # Calculating the current model score and appending to history list
            self.score_history.append(self.accuracy_score(X, y))

            # Checking for convergence
            if np.allclose(gradient, np.zeros(X_.shape[1])):
                break

    def prediction(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X)) 
        return X @ self.coefficients

    def accuracy_score(self, X, y):
        y_avg = np.mean(y)
        y_pred = self.prediction(X)
        return 1 - np.sum((y_pred - y) ** 2) / np.sum((y_avg - y) ** 2)
    
    #

