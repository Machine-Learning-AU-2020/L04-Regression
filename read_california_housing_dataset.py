# Read California Housing data
# https://scikit-learn.org/dev/datasets/index.html#california-housing-dataset
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

#%%
from sklearn.datasets.california_housing import fetch_california_housing
import matplotlib.pyplot as plt
import numpy as np

cal_housing = fetch_california_housing()
X, y = cal_housing.data, cal_housing.target
median_income = X[:, np.newaxis, 0]
median_house_value = y

plt.plot(median_income, median_house_value, '.', markersize = 1)
plt.title("Median house value for California districts relative to MedInc median income in block")
plt.xlabel("Median income")
plt.ylabel("Median house value")
plt.show()

#%%

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
#%%

residual_values = median_house_value-median_house_value_pred

plt.plot(median_income, residual_values, '.', markersize = 1)
plt.title("Residual Plot")
plt.xlabel("Median income")
plt.ylabel("Residual")
plt.show()

#%%

plt.hist(residual_values)
plt.title("Histogram of residual values")
plt.xlabel("Residual")
plt.ylabel("Median income")
plt.show()

#%%

# Train the model using the training sets
regr.fit(X, median_house_value)

# Make predictions using the testing set
median_house_value_pred = regr.predict(X)

# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(median_house_value, median_house_value_pred))

# %%
