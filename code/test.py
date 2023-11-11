import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from icecream import ic
import pandas as pd


class LinearRegression():
    def __init__(self, n=100, p=1, sigma=2) -> None:
        self.n = n
        self.p = p
        self.sigma = sigma

        self.weights = None
        self.bias = 0
        self.true_weights = None
        self.true_bias = 0
        self.true_y = None
        self.X = None
        self.pred = None
        self.residuals = None
        self.sigma_naive = None
        self.sigma_cor = None
        self.intercept = 0

    def fit(self, X, y):
        if self.p == 1:  # univariate
            xmean = np.mean(X)
            ymean = np.mean(y)

            # slope
            bhat1 = np.sum((X - xmean)*(y - ymean)) / \
                np.sum(np.square(X - xmean))
            self.weights = bhat1

            # intercept
            self.intercept = ymean - xmean * self.weights
        else:  # multi variate case
            self.weights = np.linalg.inv(
                X.T @ X) @ X.T @ y

    def predict(self, X):
        self.pred = np.dot(X, self.weights) + self.intercept

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
        plt.scatter(self.X, self.true_y,
                    label=f"Data Points", c='b', s=1)
        plt.xlabel("sqft_living")
        plt.ylabel("Housing Price")
        plt.title("Simple Linear Regression")

        # get true line
        # line = np.dot(self.X, self.true_weights)
        # plt.plot(self.X, line, label=f"True Line", color='r')

        # get model weights and bais
        self.fit(self.X, self.true_y)
        self.predict(self.X)
        plt.plot(self.X, self.pred, label=f"Regression line", color='g')

        plt.legend()
        plt.savefig("figure/simple_regression.png")
        if show:
            plt.show()
            plt.close()

    def plot_multi_data(self, generated_data=False, show=False):
        # 2 covariates only
        # Create a 3D scatter plot to visualize the data
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if generated_data:
            ax.scatter(self.X[:, 0], self.X[:, 1],
                       self.true_y, c='b', marker='o', s=1)
        else:
            ax.scatter(self.X['sqft_living'], self.X['yr_built'],
                       self.true_y, c='b', marker='o', s=1)

        self.fit(self.X, self.true_y)
        self.predict(self.X)

        if generated_data:
            x_plane = np.linspace(
                self.X[:, 0].min(), self.X[:, 0].max(), 100)
            y_plane = np.linspace(
                self.X[:, 1].min(), self.X[:, 1].max(), 100)
        else:
            x_plane = np.linspace(
                self.X['sqft_living'].min(), self.X['sqft_living'].max(), 100)
            y_plane = np.linspace(
                self.X['yr_built'].min(), self.X['yr_built'].max(), 100)
        xx, yy = np.meshgrid(x_plane, y_plane)
        z_plane = self.weights[0] * xx + self.weights[1] * yy

        # plot regression plane
        ax.plot_surface(xx, yy, z_plane, alpha=0.5,
                        color='g', label='Fitted Plane')

        ax.set_xlabel('sqft_living')
        ax.set_ylabel('yr_built')
        ax.set_zlabel('Housing Price')

        plt.title("Bi-Variate Linear Regression Data")
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


# model = LinearRegression()
# model.generate_data()
# model.fit(model.X, model.true_y)
# # ic(model.weights)
# model.plot_simple_data()
# model.plot_simple_regression_line()
# model.get_sigma_naive()
# # print(f"sigma naive: {model.sigma_naive}")

# # --------
# model2 = LinearRegression(p=2)
# model2.generate_data()
# model2.plot_multi_data(generated_data=True)


# get dataframe
df = pd.read_csv('house.csv')
# ic(df.head())
condition = df['price'] > 2000000
df = df.drop(df[condition].index)

# remove unnecessary columns
to_remove = ['id', 'date', 'waterfront', 'view',
             'condition', 'zipcode', 'lat', 'long']
data_multi = df.drop(columns=to_remove)

# target variable
target = df['price']

# univariate case
data_simple = df['sqft_living']
model_simple = LinearRegression(n=len(data_simple))
model_simple.X = data_simple
model_simple.true_y = target
model_simple.fit(data_simple, target)
model_simple.plot_simple_regression_line()
# ic(model_simple.weights)

# bi variate case
data_multi2 = df[['sqft_living', 'yr_built']]
model_multi2 = LinearRegression(n=len(data_multi2), p=len(data_multi2.columns))
model_multi2.X = data_multi2
model_multi2.true_y = target
model_multi2.fit(data_multi2, target)
model_multi2.plot_multi_data()

# multi variate case
