"""
AUTHORS:
Ling Fei Zhang, 260985358
Sevag Baghdassarian, 260980928

PROJECT:
Math 533 Final Project
"""

import numpy as np
from scipy import stats
from scipy.stats import f
from scipy.stats import t
import matplotlib.pyplot as plt
from prettytable import PrettyTable, MSWORD_FRIENDLY
import pandas as pd
import seaborn as sns
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

        # if X is a 1d array, reshape to column
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # add column of ones to X for intercept
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # compute weights
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):

        # if X is a 1d array, reshape to column
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # add column on ones to x to include intercept
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # get predictions (includes intercept)
        self.pred = np.dot(X, self.weights)

        # update sample residuals
        self.sample_residuals()

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

    def plot_simple_regression_line(self, show=False, ci=True, interval=95):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.X, self.true_y,
                    label=f"Data Points", c='b', s=1)

        # if want to show CI, default to 95
        if ci:
            df = pd.DataFrame(
                {'sqft_living': self.X, 'Housing Price': self.true_y})

            sns.regplot(x='sqft_living', y='Housing Price',
                        data=df, ci=interval, color='0.5', line_kws=dict(color='g', label=f'{interval}% CI Regression Line'), scatter_kws=dict(s=1, label=f'Data Points', color='b'))

        else:
            plt.xlabel("sqft_living")
            plt.ylabel("Housing Price")
            plt.plot(self.X, self.pred, label=f"Regression Line", color='g')

        plt.title("Simple Linear Regression")
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
        z_plane = self.weights[1] * xx + self.weights[2] * yy + self.weights[0]

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

        # update residuals
        self.sample_residuals()

        # if univariate (X is a vector), reshape to 2d array
        X_numpy = self.X.to_numpy()

        if len(X_numpy.shape) == 1:
            X_numpy = X_numpy.reshape(-1, 1)

        # get mean squared error
        MSE = np.sum(np.square(self.residuals)) / (self.n - len(self.weights))

        # add intercept term to x
        X_with_intercept = np.hstack((np.ones((X_numpy.shape[0], 1)), X_numpy))

        # get variance-covariance matrix
        var_cov_matrix = MSE * \
            np.linalg.inv(X_with_intercept.T @ X_with_intercept)

        # get standard errors (sqrt of diagonal terms)
        self.standard_errors = np.sqrt(np.diag(var_cov_matrix))

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
        row = [residuals_min, residuals_1q,
               residuals_median, residuals_3q, residuals_max]
        row = [round(item, 1) for item in row]
        residuals_table.add_row(row)
        print('Residuals:')
        residuals_table.align = 'c'
        print(residuals_table)
        table = residuals_table.get_csv_string()
        with open(f'figure/p{self.p}_residuals.csv', 'w') as file:
            file.write(table)

        # get estimates, std error, t value, Pr(>|t|) for coefficients
        coef_stats_table = PrettyTable(
            ['Coef', 'Estimate', 'Std. error', 't value', 'p value'])
        coef_stats = []
        coef_names = ['Intercept'] + [f'x{i}' for i in range(1, self.p + 1)]
        print('Coefficients:')
        for idx, weight in enumerate(self.weights):
            if self.p <= 2:
                # compute p value for weight
                p_value = 2 * \
                    (1 -
                     stats.t.cdf(abs(self.t_values[idx]), df=self.n-self.p-1))
                # format p value to not display small values as 0
                formatted_p_value = "{:.4e}".format(p_value)
                coef_stats.append([
                    coef_names[idx],
                    round(weight, 2),
                    round(self.standard_errors[idx], 2),
                    round(self.t_values[idx], 2),
                    formatted_p_value
                ])
            else:
                # compute p value for weight
                p_value = 2 * \
                    (1 -
                     stats.t.cdf(abs(self.t_values[idx]), df=self.n-self.p-1))
                # format p value to not display small values as 0
                formatted_p_value = "{:.4e}".format(p_value)
                coef_stats.append([
                    coef_names[idx],
                    f'{weight:.4e}',
                    f'{self.standard_errors[idx]:.4e}',
                    f'{self.t_values[idx]:.4e}',
                    formatted_p_value
                ])

        for stat in coef_stats:
            coef_stats_table.add_row(stat)

        print(coef_stats_table)
        table = coef_stats_table.get_csv_string()
        with open(f'figure/p{self.p}_coef_stats_table.csv', 'w') as file:
            file.write(table)

        with open(f'figure/p{self.p}_texts.tex', 'w') as file:
            # residual std error and p value
            rse = f'Residual standard error: {self.RSE:.2f} on {(self.n - self.p - 1)} degrees of freedom\n\n'
            print(rse, end='')
            file.write(rse)

            # R2 and adj R2
            r2 = f'R-squared: {self.R2:.2f}, Adjusted R-squared: {self.adj_R2:.2f}\n\n'
            print(r2, end='')
            file.write(r2)

            # F-statistic and p value
            f = f'F-statistic: {self.F_stat:.2f} on {self.p} and {self.n - self.p - 1} DF, p-value: {self.p_value}\n\n'
            print(f, end='')
            file.write(f)

        # print other metrics
        df_metrics = pd.DataFrame({
            'Sigma naive': self.sigma_naive,
            'Sigma cor': self.sigma_cor,
            'MSE': self.MSE,
            'MAE': self.MAE,
            'RMSE': self.RMSE
        }, index=[0])
        other_metrics_table = PrettyTable(
            ['Sigma naive', 'Sigma cor', 'MSE', 'MAE', 'RMSE'])
        row = [self.sigma_naive, self.sigma_cor, self.MSE, self.MAE, self.RMSE]
        row = [round(item, 1) for item in row]
        other_metrics_table.add_row(row)
        other_metrics_table.align = 'c'
        print(other_metrics_table)
        table = other_metrics_table.get_csv_string()
        with open(f'figure/p{self.p}_other_metrics_table.csv', 'w') as file:
            file.write(table)

    # helper function to get data quartiles
    def get_quartiles(self, data):
        data_cpy = np.copy(data)
        data_cpy.sort()
        q1 = np.percentile(data_cpy, 25)
        q3 = np.percentile(data_cpy, 75)
        return q1, q3

    def update_metrics(self):
        self.sample_residuals()
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

    # function to get confidence intervals for each weight given a confidence level
    # returns a dict mapping from each coefficient to a (lower bound, upper bound) tuple
    def get_confidence_intervals(self, confidence_level=0.95):

        # get degrees of freedom
        df = self.n - len(self.weights)

        # get critical value from t dist
        critical_value = t.ppf(1 - (1 - confidence_level)/2, df)

        # get confidence intervals
        confidence_intervals = {}
        coef_names = ['Intercept'] + [f'x{i}' for i in range(1, self.p + 1)]
        for idx, (coef, std_error) in enumerate(zip(self.weights, self.standard_errors)):
            # get error margin lower and upper bounds
            error_margin = critical_value * std_error
            lower_bound = coef - error_margin
            upper_bound = coef + error_margin
            confidence_intervals[coef_names[idx]] = (lower_bound, upper_bound)

        return confidence_intervals

    # function to perform hypothesis testing
    # tests null hypothesis, where H0: coefficient = 0, for each coefficient in the model
    def perform_hypothesis_testing(self, significance_level=0.05):

        # get degrees of freedom (n-p-1)
        df = self.n - len(self.weights)

        # get hypothesis test results for each coefficient
        hypothesis_test_results = []
        coef_names = ['Intercept'] + [f'x{i}' for i in range(1, self.p + 1)]
        for idx, (coef, std_error) in enumerate(zip(self.weights, self.standard_errors)):

            # get t value
            t_value = coef / std_error

            # get p value
            p_value = 2 * (1 - t.cdf(abs(t_value), df))

            # add to dict
            hypothesis_test_results.append([
                coef_names[idx],
                round(t_value, 2),
                round(p_value, 2),
                p_value < significance_level
            ]
            )

        hypothesis_test_table = PrettyTable(
            ['Coefficient', 't-value', 'p-value', 'Reject Null Hypothesis?'])
        for test in hypothesis_test_results:
            hypothesis_test_table.add_row(test)

        print('Hypothesis tests:')
        print(hypothesis_test_table)

        # send table to file
        table = hypothesis_test_table.get_csv_string()
        with open(f'figure/p{self.p}_hypothesis_tests.csv', 'w') as file:
            file.write(table)
