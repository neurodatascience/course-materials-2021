# # Getting started with supervised learning: linear regression
#
# We use a small dataset distributed with scikit-learn, containing information
# about diabetic patients. 10 features were collected: age, sex, body mass
# index, average blood pressure, and six blood serum measurements. The target
# variable to predict is a continuous value measuring the progression of
# diabetes one year after the features were collected.
#
# To predict the disease progress, we will use a linear regression --
# implemented by `sklearn.linear_model.LinearRegression`. We will compare it to
# a "dummy" model, which always predicts the same thing, regardless of the
# input features.


# +
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt

# -

# We now load the dataset in memory as 2 numpy arrays `X` and `y`. `X` has
# shape `n_samples, n_features = 442, 10`, and `y` has shape `n_samples = 442`.
#
# We split the data into training and test examples, keeping 80% for training
# and 20% for testing.

# +
X, y = datasets.load_diabetes(return_X_y=True)

n_train = int(len(y) * 0.8)
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]
# -

# We fit the model to the training data.
#
# **Exercise**: what will be the size of the model's coefficients (excluding
# the intercept)?

model = LinearRegression()
model.fit(X_train, y_train)

# **Exercise**: what function of `X_train`, `y_train`, and the coefficients
# `beta` does the linear regression minimize in order to estimate `beta`?
# Implement this function below.


def loss_function(X, y, beta):
    pass


# Now we evaluate our model on (unseen) test data and display its Mean Squared
# Error.

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean squared error on test data: {mse:.2g}")

# We train a `DummyRegressor`. This estimator makes a constant prediction (it
# ignores the features and always predicts the same value for `y`). However,
# this constant value is not arbitrary: it is the one that results in the
# smallest Mean Squared Error on the training data.
#
# **Exercise**: what constant value minimizes the MSE for the training sample?

dummy_predictions = DummyRegressor().fit(X_train, y_train).predict(X_test)
dummy_mse = mean_squared_error(y_test, dummy_predictions)
print(f"Mean squared error of dummy model on test data: {dummy_mse:.2g}")

# Finally, we display the true outcomes and the predictions of our models for
# the test data. Would you say the linear regression is doing much better than
# the dummy model?

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="black",
    linestyle="--",
)
plt.scatter(y_test, predictions)
plt.scatter(y_test, dummy_predictions, marker="^")
plt.legend(
    [
        "Perfect prediction",
        f"LinearRegression (MSE = {mse:.2g})",
        f"DummyRegressor (MSE = {dummy_mse:.2g})",
    ]
)
plt.gca().set_xlabel("True outcome")
plt.gca().set_ylabel("Predicted outcome")
plt.gca().set_title("True and predicted diabetes progress")
plt.show()

# Here we have a small number of features that are not too correlated
# (condition number of `X_train` is 23), so linear regression without
# regularization works well. If the number of features were large, or if the
# columns of X were not linearly independent, what could we use to stabilize
# the model's parameters?
