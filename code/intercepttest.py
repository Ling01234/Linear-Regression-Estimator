
import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from icecream import ic
from prettytable import PrettyTable
import pandas as pd
from regression import LinearRegression

# get dataframe
df = pd.read_csv('house.csv')
data_simple = df

# target variable
target = df['price']

# univariate case
data_simple = df['sqft_living']
X = data_simple.to_numpy()
model_simple = LinearRegression(n=len(data_simple))
model_simple.X = X
model_simple.true_y = target
model_simple.fit(model_simple.X, model_simple.true_y)
model_simple.predict(model_simple.X)
model_simple.summarize()
# model_simple.perform_hypothesis_testing()

# univariate case
# data_simple = df['sqft_living']
# X = data_simple.to_numpy()
# model_simple = LinearRegression(n=len(data_simple))
# model_simple.X = X
# model_simple.true_y = target
# model_simple.fit(model_simple.X, target)
# model_simple.predict(model_simple.X)
# # model_simple.plot_simple_regression_line()
# model_simple.summarize()
# model_simple.perform_hypothesis_testing()