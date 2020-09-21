import numpy as np


class Perceptron:
    def __init__(self):
        self.nx = self.W = self.b = None


    def sigmoid(self, z):
        """Implements sigmoid function"""
        return 1/(1+np.exp(-z))


    def acc(self, X, Y):
        """Calculates accuracy given labels"""
        A = self.predict(X) > .5
        return np.mean(A == Y)


    def XC(self, X, Y, lambd):
        """Calculate cross-entropy loss with regularization"""
        nx, m = X.shape
        assert(nx == self.nx)
        A = self.predict(X)
        loss = -np.mean(Y*np.log(A) + (1-Y)*np.log(1-A))
        regular = lambd/(2*m) * np.square(self.W).sum()
        return loss + regular


    def predict(self, X):
        """Implements prediction using existing weights and biases"""
        preds = self.sigmoid(self.W.T @ X + self.b)
        return np.squeeze(preds)
    

    def grad(self, X, Y):
        """Calculates gradients"""
        nx, m = X.shape
        assert(nx == self.nx)
        A = self.predict(X)
        dZ = A - Y
        db = np.mean(dZ)
        dW = (X @ dZ.T) / m
        dW = dW.reshape((self.nx,1))
        return dW, db
    

    def fit(self, X, Y, alpha, lambd, epochs):
        """Fits weights and biases to dataset"""
        nx, m = X.shape

        self.nx = nx
        self.W = np.zeros((nx,1))
        self.b = 0
        self.cost = None

        for i in range(epochs):
            dW, db = self.grad(X, Y)
            self.W -= alpha * (dW + lambd/(2*m)*self.W)
            self.b -= alpha * db
            if i % 10_000 == 0:
                print("Cost for epoch %d is %.5f" %
                    (i, self.XC(X,Y,lambd)))