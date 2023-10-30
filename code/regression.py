import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class LinearRegression():
    def __init__(self, n=100, p=1, sigma=2) -> None:
        self.n = n
        self.p = p
        self.sigma = sigma

        self.weights = None
        self.bias = None
        self.true_weights = None
        self.true_bias = None
        self.true_y = None
        self.X = None
        self.pred = None
        self.residuals = None
        self.sigma_naive = None
        self.sigma_cor = None

    def get_bhat(self):
        bhat = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        self.weights = bhat
        # return bhat

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

    def generate_data(self):
        if self.p == 1:
            self.X = np.arange(self.n)
            self.true_weights = np.random.uniform()
        else:
            self.X = np.random.rand(self.n, self.p) * 10
            self.true_weights = np.random.uniform(0, 10, self.p)
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
            plt.close()

    def plot_simple_regression_line(self, show=False):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.X, self.true_y, label=f"Data Points", c='b')
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Simple Linear Regression")

        # get true line
        line = np.dot(self.X, self.true_weights)
        plt.plot(self.X, line, label=f"True Line", color='r')

        # get model weights and bais
        self.simple_bhat1()
        self.simple_bhat0()
        self.pred = np.dot(self.X, self.weights) + self.bias
        plt.plot(self.X, self.pred, label=f"Regression line", color='g')

        plt.legend()
        plt.savefig("figure/simple_regression.png")
        if show:
            plt.show()
            plt.close()

    def plot_multi_data(self, show=False):
        # 2 covariates only
        # Create a 3D scatter plot to visualize the data
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.X[:, 0], self.X[:, 1], self.true_y, c='b', marker='o')

        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Y')

        plt.title("Multiple Linear Regression Data")
        plt.savefig("figure/multi_data.png")
        if show:
            plt.show()
            plt.close()

    def sample_residuals(self):
        self.residuals = self.true_y - self.pred

    def get_sigma_naive(self):
        self.sample_residuals()
        sum = np.sum(np.square(self.residuals))
        self.sigma_naive = sum/self.n

    def get_sigma_cor(self):
        self.sigma_cor = 1/(self.n - self.p) * \
            np.sum(np.square(self.residuals))


model = LinearRegression()
model.generate_data()
model.plot_simple_data()
model.plot_simple_regression_line()
model.get_sigma_naive()
# print(f"sigma naive: {model.sigma_naive}")

# --------
model2 = LinearRegression(p=2)
model2.generate_data()
model2.plot_multi_data()

