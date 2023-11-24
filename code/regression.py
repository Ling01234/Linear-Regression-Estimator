import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from icecream import ic
from prettytable import PrettyTable
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


    def get_R2(self):
        ymean = np.mean(self.true_y)
        SSR = np.sum(np.square(self.true_y - self.pred))
        SST = np.sum(np.square(self.true_y - ymean))
        self.R2 = 1 - SSR/SST

    def get_adj_R2(self):
        self.get_R2()
        fraction = (1 - self.R2)*(self.n - 1) / (self.n - self.p - 1)
        self.adj_R2 = 1 - fraction

    def get_MSE(self):
        sum_of_squares = np.sum(np.square(self.true_y - self.pred))
        self.MSE = sum_of_squares/self.n

    # MAE: Mean Absolute Error
    def get_MAE(self):
        sum_of_abs = np.sum(np.absolute(self.true_y - self.pred))
        self.MAE = sum_of_abs/self.n

    # RMSE: Root Mean Squared Error
    def get_RMSE(self):
        self.get_MSE()
        self.RMSE = np.sqrt(self.MSE)

    # RSE: Residual Standard Error
    def get_RSE(self):
        sum_sq_res = np.sum(np.square(self.residuals))
        self.RSE = np.sqrt(sum_sq_res / (self.n - self.p - 1))

    def get_F_stat(self):
        ymean = np.mean(self.true_y)
        SST = np.sum(np.square(self.true_y - ymean))
        SSR = np.sum(np.square(self.pred - ymean))
        RSS = np.sum(np.square(self.true_y - self.pred))
        MSR = SSR / self.p
        MSE = RSS / (self.n - self.p - 1)
        self.F_stat = MSR / MSE

    def get_p_value(self):
        self.p_value = 1 - f.cdf(self.F_stat, self.p, self.n - self.p - 1)

    def get_standard_errors(self):
        RSS = np.sum(np.square(self.residuals))
        MSE = RSS / (self.n - self.p - 1)
        # XtX_inverse = np.linalg.inv(self.X.T.dot(self.X))
        if self.p != 1:
            XtX_inverse = np.linalg.inv(np.dot(self.X.T, self.X))
            self.standard_errors = np.sqrt(MSE * np.diag(XtX_inverse))
        else:
            xmean = np.mean(self.X)
            self.standard_errors = MSE / np.sum(np.square(self.X - xmean))

    def get_t_values(self):
        if self.standard_errors is not None:
            self.t_values = self.weights / self.standard_errors
            
    # function to print a pretty table containing values
    def print_pretty_table(headers, rows):
        pretty_table = PrettyTable(headers)
        for row in rows:
            pretty_table.add_row(row)
            
        print(pretty_table)
        
    def summarize(self):

        if self.weights is None:
            raise ValueError("Linear regression has not yet been performed.")

        self.update_metrics()

        # print residuals info
        residuals_1q, residuals_3q = self.get_quartiles(self.residuals)
        residuals_min = np.min(self.residuals)
        residuals_median = np.median(self.residuals)
        residuals_max = np.max(self.residuals)
        residuals_table = PrettyTable(['Min', '1Q', 'Median', '3Q', 'Max'])
        residuals_table.add_row([residuals_min, residuals_1q, residuals_median, residuals_3q, residuals_max])
        print('Residuals:')
        print(residuals_table)

        # get estimates, std error, t value, Pr(>|t|) for coefficients
        coef_stats = []
        if self.p != 1:
            for idx, weight in enumerate(self.weights):
                coef_stats.append({
                    'Coef': f'x{idx}',
                    'Estimate': weight,
                    'Std. error': self.standard_errors[idx],
                    't value': self.t_values[idx],
                    'p value': self.p_value # might need to change
                })
        else:
            coef_stats.append({
                'Coef': f'x{0}',
                'Estimate': self.weights,
                'Std. error': self.standard_errors,
                't value': self.t_values,
                'p value': self.p_value
            })
        
        # print coef stats
        coef_stats_table = PrettyTable(['Coef', 'Estimate', 'Std. error', 't value', 'p value'])
        print('Coefficients:')
        for stat in coef_stats:
            coef_stats_table.add_row([stat['Coef'], stat['Estimate'], stat['Std. error'], stat['t value'], stat['p value']])
        print(coef_stats_table)
        
        # residual std error and p value
        print(f'Residual standard error: {self.RSE} on {self.n - self.p - 1} degrees of freedom')

        # R2 and adj R2
        print(f'R-squared: {self.R2}, Adjusted R-squared: {self.adj_R2}')

        # F-statistic and p value
        print(
            f'F-statistic: {self.F_stat} on {self.p} and {self.n - self.p - 1} DF, p-value: {self.p_value}')

        # print other metrics
        df_metrics = pd.DataFrame({
            'Sigma naive': self.sigma_naive,
            'Sigma cor': self.sigma_cor,
            'MSE': self.MSE,
            'MAE': self.MAE,
            'RMSE': self.RMSE
        }, index=[0])
        other_metrics_table = PrettyTable(['Sigma naive', 'Sigma cor', 'MSE', 'MAE', 'RMSE'])
        other_metrics_table.add_row([self.sigma_naive, self.sigma_cor, self.MSE, self.MAE, self.RMSE])
        print(other_metrics_table)

    # helper function to get data quartiles
    def get_quartiles(self, data):
        data_cpy = np.copy(data)
        data_cpy.sort()
        q1 = np.percentile(data_cpy, 25)
        q3 = np.percentile(data_cpy, 75)
        return q1, q3

    def update_metrics(self):
        self.get_sigma_naive()
        self.get_sigma_cor()
        self.get_R2()
        self.get_adj_R2()
        self.get_MSE()
        self.get_MAE()
        self.get_RMSE()
        self.get_RSE()
        self.get_F_stat()
        self.get_p_value()
        self.get_standard_errors()
        self.get_t_values()