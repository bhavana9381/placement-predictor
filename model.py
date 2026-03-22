import math

class LogisticRegressionMulti:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = []
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def train(self, X, Y):
        n = len(X)
        features = len(X[0])

        self.weights = [0] * features
        self.bias = 0

        for _ in range(self.epochs):
            dw = [0] * features
            db = 0

            for i in range(n):
                z = sum(self.weights[j] * X[i][j] for j in range(features)) + self.bias
                y_pred = self.sigmoid(z)

                for j in range(features):
                    dw[j] += (y_pred - Y[i]) * X[i][j]

                db += (y_pred - Y[i])

            for j in range(features):
                self.weights[j] -= self.lr * dw[j] / n

            self.bias -= self.lr * db / n

    def predict_proba(self, x):
        z = sum(self.weights[j] * x[j] for j in range(len(x))) + self.bias
        return self.sigmoid(z)

    def predict(self, x):
        return 1 if self.predict_proba(x) >= 0.5 else 0