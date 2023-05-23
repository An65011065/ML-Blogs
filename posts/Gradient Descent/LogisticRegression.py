import numpy as np

class LogisticRegression:
    def __init__(self):
        self.w = None
        self.loss_history = []
        self.score_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, alpha=0.1, max_epochs=1000):
        n, m = X.shape
        self.w = np.zeros(m+1)
        X_bias = np.insert(X, 0, 1, axis=1)

        for epoch in range(max_epochs):
            z = np.dot(X_bias, self.w)
            h = self.sigmoid(z)
            gradient = np.dot(X_bias.T, (h - y)) / y.size
            self.w -= alpha * gradient

            loss = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
            self.loss_history.append(loss)

            score = self.score(X, y)
            self.score_history=self.score_history.append(score)

    def fit_stochastic(self, X, y, alpha=0.1, max_epochs=1000, batch_size=32):
        n, m = X.shape
        self.w = np.zeros(m+1)
        X_bias = np.insert(X, 0, 1, axis=1)
        
        for epoch in range(max_epochs):
            order = np.random.permutation(n)
            X_shuffled = X_bias[order]
            y_shuffled = y[order]
            
            for i in range(0, n, batch_size):
                xi = X_shuffled[i:i+batch_size]
                yi = y_shuffled[i:i+batch_size]
                
                z = np.dot(xi, self.w)
                h = self.sigmoid(z)
                gradient = np.dot(xi.T, (h - yi)) / batch_size
                self.w -= alpha * gradient
                
                loss = (-yi * np.log(h) - (1 - yi) * np.log(1 - h)).mean()
                self.loss_history.append(loss)

                score = self.score(X, y)
                self.score_history.append(score)
            
    def predict(self, X):
        X_bias = np.insert(X, 0, 1, axis=1)
        return self.sigmoid(np.dot(X_bias, self.w)).round()

    def score(self, X, y):
        return (self.predict(X) == y).mean()

    def loss(self, X, y):
        X_bias = np.insert(X, 0, 1, axis=1)
        h = self.sigmoid(np.dot(X_bias, self.w))
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
