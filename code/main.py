
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

# target variable
target = df['price']

# univariate case
data_simple = df['sqft_living']
model_simple = LinearRegression(n=len(data_simple))
model_simple.X = data_simple
model_simple.true_y = target
model_simple.fit(model_simple.X.to_numpy(), target)
model_simple.predict(model_simple.X.to_numpy())
model_simple.summarize()
model_simple.perform_hypothesis_testing()
model_simple.plot_simple_regression_line()
# ic(model_simple.weights)

# bi variate case
data_multi2 = df[['sqft_living', 'yr_built']]
model_multi2 = LinearRegression(n=len(data_multi2), p=len(data_multi2.columns))
model_multi2.X = data_multi2
model_multi2.true_y = target
model_multi2.fit(model_multi2.X.to_numpy(), target)
model_multi2.predict(model_multi2.X.to_numpy())
model_multi2.summarize()
model_multi2.perform_hypothesis_testing()
model_multi2.plot_multi_data()

# multi varaite case
# remove unnecessary columns
to_remove = ['id', 'date', 'waterfront', 'view',
             'condition', 'zipcode', 'lat', 'long']
data_multi = df.drop(columns=to_remove)
model_multi = LinearRegression(n=len(data_multi), p=len(data_multi.columns))
model_multi.X = data_multi
model_multi.true_y = target
model_multi.fit(model_multi.X.to_numpy(), target)
model_multi.predict(model_multi.X.to_numpy())
model_multi.summarize()
model_multi.perform_hypothesis_testing()