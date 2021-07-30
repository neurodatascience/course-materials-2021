import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Question: what is the issue in the code below?

# +
X, y = make_regression(n_samples=80, n_features=600, noise=10, random_state=0)

model = Ridge(alpha=1e-8)
model.fit(X, y)
predictions = model.predict(X)
mse = mean_squared_error(y, predictions)

print(f"\nMean Squared Error: {mse}")
print("MSE is 0 up to machine precision:", np.allclose(mse, 0))
# -

# Let's compare training and testing performance

scores = cross_validate(
    model, X, y, return_train_score=True, scoring="neg_mean_squared_error"
)

# Question: what CV strategy are we using? what would be the default `scoring`
# if we did not specify it?

# +
errors = -pd.DataFrame(scores)
errors = (
    errors.loc[:, ["train_score", "test_score"]]
    .rename(
        columns={
            "train_score": "Training error",
            "test_score": "Testing error",
        }
    )
    .stack()
    .reset_index()
)
errors.columns = ["split", "data", "score"]

sns.stripplot(data=errors, x="score", y="data")
plt.gca().set_xlabel("")
plt.gca().set_ylabel("")
plt.gca().set_title("Mean Squared Error")
plt.tight_layout()
plt.show()
# -
