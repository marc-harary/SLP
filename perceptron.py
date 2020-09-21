import numpy as np


class Perceptron:
    def __init__(self):
        self.nx = self.W = self.b = None


    def sigmoid(self, z):
        "Test"
        return 1/(1+np.exp(-z))


    def acc(self, X, Y):
        A = self.predict(X) > .5
        return np.mean(A == Y)


    def XC(self, X, Y, lambd):
        _, m = X.shape
        A = self.predict(X)
        loss = -np.mean(Y*np.log(A) + (1-Y)*np.log(1-A))
        regular = lambd/(2*m) * np.square(self.W).sum()
        return loss + regular


    def predict(self, X):
        preds = self.sigmoid(self.W.T @ X + self.b)
        return np.squeeze(preds)
    

    def grad(self, X, Y):
        nx, m = X.shape
        A = self.predict(X)
        dZ = A - Y
        dW = (X @ dZ.T) / m
        db = np.mean(dZ)
        return dW.reshape((nx,1)), db
    

    def fit(self, X, Y, alpha, lambd, epochs):
        nx, m = X.shape

        self.nx = nx
        self.W = np.zeros((nx,1))
        self.b = 0
        self.cost = None

        for i in range(epochs):
            dW, db = self.grad(X, Y)
            self.W -= alpha * (dW + lambd/(2*m)*self.W)
            self.b -= alpha * db
            if i % 1000 == 0:
                print("Cost for epoch %d is %.5f" %
                    (i, self.XC(X,Y,lambd)))