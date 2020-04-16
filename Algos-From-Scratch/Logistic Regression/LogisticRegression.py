# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preparation
mean_01 = np.array([1, 0.5])
cov_01 = np.array([[1, 0.1], [0.1, 1.2]])

mean_02 = np.array([4, 5])
cov_02 = np.array([[1.21, 0.1], [0.1, 1.3]])

# Normal Distribution
dist_01 = np.random.multivariate_normal(mean_01, cov_01, 500)
dist_02 = np.random.multivariate_normal(mean_02, cov_02, 500)

print(dist_01.shape)

# Data Visualise
plt.style.use('seaborn')
plt.figure(0)
plt.scatter(dist_01[:, 0], dist_01[:, 1], label='Class 0')
plt.scatter(dist_02[:, 0], dist_02[:, 1], color='r', marker='^', label='Class 1')
plt.xlim(-5, 10)
plt.ylim(-5, 10)
plt.xlabel('x1')
plt.ylabel('y1')
plt.legend()
plt.show()

# Logistic Regression Functions
def hypothesis():
    
