import numpy as np
import matplotlib.pyplot as plt


class LinearRegression():
    def __init__(self, simple=True) -> None:
        self.weights = None
        self.bias = None
        self.simple = simple

    def get_bhat(self, X, y):
        bhat = np.linalg.inv(X.T @ X) @ X.T @ y
        return bhat

    def simple_bhat1(self, X, y):
        xmean = np.mean(X)
        ymean = np.mean(y)

        bhat1 = np.sum((X - xmean)*(y - ymean))/np.sum(np.square(X - xmean))
        return bhat1

    def simple_bhat0(self, X, y):
        xmean = np.mean(X)
        ymean = np.mean(y)
        bhat1 = self.simple_bhat1(X, y)

        return ymean - xmean * bhat1

    def generate_simple_data(self, n, sigma):
        # y = X * beta + eps
        # X = np.random.rand(n, p)
        X = np.arange(n)
        beta = np.random.rand()
        eps = np.random.normal(0, np.sqrt(sigma), n)
        y = np.dot(X, beta) + eps

        return X, y, beta, eps

    def plot_simple_data(self, X, y, beta, eps, show=False):
        plt.figure(figsize=(8, 6))
        plt.scatter(X, y, label=f"Data Points", color='b')
        line = np.dot(X, beta)
        plt.plot(X, line, label=f"True Line", color='r')
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Data Points")
        plt.legend()

        plt.savefig("figure/simple_data.png")
        if show:
            plt.show()

    def plot_simple_regression_line(self, X, y, beta, eps, show=False):
        plt.figure(figsize=(8, 6))
        plt.scatter(X, y, label=f"Data Points", c='b')
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Simple Linear Regression")

        true_y = np.dot(X, beta)
        plt.plot(X, true_y, label=f"True Line", color='r')

        bhat1 = self.simple_bhat1(X, y)
        bhat0 = self.simple_bhat0(X, y)
        pred = np.dot(X, bhat1) + bhat0
        plt.plot(X, pred, label=f"Regression line", color='g')

        plt.legend()
        plt.savefig("figure/simple_regression.png")
        if show:
            plt.show()


model = LinearRegression()
X, y, beta, eps = model.generate_simple_data(100, 2)
model.plot_simple_data(X, y, beta, eps)
model.plot_simple_regression_line(X, y, beta, eps)
