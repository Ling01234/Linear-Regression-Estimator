import numpy as np
import matplotlib.pyplot as plt


class LinearRegression():
    def __init__(self, simple=True, n=100, p=1, sigma=2) -> None:
        self.simple = simple

        self.weights = None
        self.bias = None
        self.true_weights = None
        self.true_bias = None
        self.true_y = None
        self.X = None
        self.n = n
        self.p = p
        self.sigma = sigma

    def get_bhat(self):
        bhat = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        return bhat

    def simple_bhat1(self):
        xmean = np.mean(self.X)
        ymean = np.mean(self.true_y)

        bhat1 = np.sum((self.X - xmean)*(self.true_y - ymean)) / \
            np.sum(np.square(self.X - xmean))
        self.weights = bhat1
        # return bhat1

    def simple_bhat0(self):
        xmean = np.mean(self.X)
        ymean = np.mean(self.true_y)

        self.bias = ymean - xmean * self.weights
        # return ymean - xmean * bhat1

    def generate_simple_data(self):
        self.X = np.arange(self.n)
        self.true_weights = np.random.rand()  # p = 1
        self.true_bias = np.random.normal(0, np.sqrt(self.sigma), self.n)
        self.true_y = np.dot(self.X, self.true_weights) + self.true_bias

    def plot_simple_data(self, show=False):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.X, self.true_y, label=f"Data Points", color='b')
        line = np.dot(self.X, self.true_weights)
        plt.plot(self.X, line, label=f"True Line", color='r')
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Data Points")
        plt.legend()

        plt.savefig("figure/simple_data.png")
        if show:
            plt.show()

    def plot_simple_regression_line(self, show=False):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.X, self.true_y, label=f"Data Points", c='b')
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Simple Linear Regression")

        line = np.dot(self.X, self.true_weights)
        plt.plot(self.X, line, label=f"True Line", color='r')

        self.simple_bhat1()
        self.simple_bhat0()
        pred = np.dot(self.X, self.weights) + self.bias
        plt.plot(self.X, pred, label=f"Regression line", color='g')

        plt.legend()
        plt.savefig("figure/simple_regression.png")
        if show:
            plt.show()


model = LinearRegression(n=100, sigma=2)
model.generate_simple_data()
model.plot_simple_data()
model.plot_simple_regression_line()
