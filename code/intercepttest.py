
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
data_simple = df[['sqft_living', 'zipcode']]
# data_simple = df['sqft_living']
X = data_simple.to_numpy()
model_simple = LinearRegression(n=len(data_simple), p=2)
model_simple.X = X
model_simple.true_y = target
model_simple.fit(model_simple.X, model_simple.true_y)
model_simple.predict(model_simple.X)
model_simple.summarize()


# # bi variate case
# data_multi2 = df[['sqft_living', 'yr_built']]
# model_multi2 = LinearRegression(n=len(data_multi2), p=len(data_multi2.columns))
# model_multi2.X = data_multi2
# model_multi2.true_y = target
# model_multi2.fit(data_multi2, target)
# model_multi2.plot_multi_data()