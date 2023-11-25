
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
