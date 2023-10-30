import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set a random seed for reproducibility
np.random.seed(0)

# Number of data points
n = 100

# Generate random covariates x1 and x2
x1 = np.random.rand(n) * 10
x2 = np.random.rand(n) * 5

# Generate random coefficients
beta1 = 2.0
beta2 = 3.0
beta0 = 5.0

# Generate random noise
sd = 2.0
eps = np.random.normal(0, sd, n)

# Generate the target variable y using a multiple linear regression model
y = beta0 + beta1 * x1 + beta2 * x2 + eps

# Create a 3D scatter plot to visualize the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x1, x2, y, c='b', marker='o')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')

plt.title("Multiple Linear Regression Data")

plt.show()
