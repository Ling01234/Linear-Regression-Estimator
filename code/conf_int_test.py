
import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from icecream import ic
from prettytable import PrettyTable
import pandas as pd
from regression import LinearRegression

df = pd.read_csv('house.csv')

y = df['price']
X = df['sqft_living']
X = X.to_numpy()

lr = LinearRegression(n=len(X), p=1)
lr.X = X
lr.true_y = y
lr.fit(X, y)
lr.predict(X)
lr.summarize()