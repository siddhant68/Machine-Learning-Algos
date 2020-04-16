import numpy as np

# Getting Multivariate Dataset
from sklearn.datasets import load_boston
boston = load_boston()

X = boston.data
Y = boston.target

# Splitting Dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Muliple Linear Regression to dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(normalize=True)
regressor.fit(x_train, y_train)
print(regressor.coef_, regressor.intercept_)

# Predicting Results
y_pred = regressor.predict(x_test)

# Accuracy Percentage
acc = (np.sum(y_pred)/ np.sum(y_test)) * 100
print(regressor.score(x_train, y_train))
print(regressor.score(x_test, y_test))
