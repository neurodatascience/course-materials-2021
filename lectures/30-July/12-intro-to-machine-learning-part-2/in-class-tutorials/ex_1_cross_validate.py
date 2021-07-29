from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge

X, y = make_regression(noise=10)
model = Ridge()

# TODO: using an appropriate function from scikit-learn, compute
# cross-validation scores for a ridge regression on this dataset. What
# cross-validation strategy is used? what do the scores represent -- what
# performance metric is used?
scores = "???"
print(scores)
