import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class Perceptron:
    def __init__(self):
        self.w = None
        self.history = []
        
    def fit(self, X, y, max_steps):
        # X: matrix of n items and p features
        # y: binary labels (-1 or 1)
        # max_steps: maximum number of iterations
        
        #Assigning the number of rows in X to M and the number of columns to N
        m, n = X.shape
        
        # Adding a column of ones to X to learn bias term.
        X_ = np.hstack([X, np.ones((m, 1))])
        
        # Initializing weights randomly
        self.w = np.random.rand(n+1)
        
        # Looping over max_steps iterations, selecting random datapoints
        for i in range(max_steps):
            i = np.random.randint(m)
            
            #Converting binary labels to -1 or 1
            y_tilde = 2*y-1
            
            # Checking if the current datapoint is misclassified
            if (y_tilde[i] * np.dot(X_[i], self.w)) <= 0:
                # If misclassified, update weights
                self.w += y_tilde[i] * X_[i]
                
            # Calculating accuracy on current iteration and updating on history
            accuracy = self.score(X, y)
            self.history.append(accuracy)
            
            # terminate the loop if accuracy is 1
            if accuracy == 1.0:
                break
            
        # append final accuracy score to history
        accuracy = self.score(X, y)
        self.history.append(accuracy)
            
    def predict(self, X):
        #performs a dot product between the input matrix X and the learned weights, and returns binary predictions based on the sign of the dot product
        
        
        # Adding a column of ones to X to learn bias term.
        X_ = np.hstack([X, np.ones((X.shape[0], 1))])
        
        # Predicting binary labels(0,1)  based on sign of dot product
        #Dot product of matrix X_ and the weight vector self.w
        return np.where(np.dot(X_, self.w) >= 0, 1, 0)

    def score(self, X, y):
        # Makeing binary labels for X and returning mean accuracy
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
