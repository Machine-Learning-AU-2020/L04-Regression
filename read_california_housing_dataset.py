# Read California Housing data
# https://scikit-learn.org/dev/datasets/index.html#california-housing-dataset
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

from sklearn.datasets.california_housing import fetch_california_housing
import matplotlib.pyplot as plt

cal_housing = fetch_california_housing()
X, y = cal_housing.data, cal_housing.target
names = cal_housing.feature_names

#%% Show example data

Nsamples = X.shape[0]

n = 42
xn = X[n, :]

# Input values
for i,val in enumerate(xn):
    print(names[i], val)

# Target value
print('Target :', y[n])

#%% 1D data plot

plt.plot(X[:,0], y, '.', markersize=1)
plt.title(names[0])


# %%
#%% New names for variables
import numpy as np
median_income = X[:, np.newaxis, 0]
median_house_value = y

plt.plot(median_income, median_house_value, '.', markersize = 1)
plt.title("Median house value for California districts i forhold til MedInc median income in block")
plt.xlabel("Median income")
plt.ylabel("Median house value")
plt.show()

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(median_income, median_house_value)

# Make predictions using the testing set
median_house_value_pred = regr.predict(median_income)

# The coefficients
print('Coefficients: \n', regr.coef_)

# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(median_house_value, median_house_value_pred))
# The coefficient of determination: 1 is perfect prediction

plt.show()

residual_values = median_house_value-median_house_value_pred

plt.plot(median_income, residual_values, '.', markersize = 1)
plt.title("residual-værdierne som funktion af feature-værdi")
plt.xlabel("Median income")
plt.ylabel("Residual værdier")
plt.show()

plt.hist(residual_values)
plt.show()

# Train the model using the training sets
regr.fit(X, median_house_value)

# Make predictions using the testing set
median_house_value_pred = regr.predict(X)

# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(median_house_value, median_house_value_pred))