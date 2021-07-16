# # Performing nested cross-validation
#
# Here we will evaluate the performance of an L2-regularized logistic
# regression on one of the classification datasets distributed with
# scikit-learn.
#
# The model has a hyperparameter C, which controls the regularization strength:
# a higher value of C means less regularization. We will automatically select
# the appropriate value for C among a grid of possible values with a nested
# cross-validation loop.
#
# The whole procedure therefore looks like:
#
# - Outer loop:
#   - obtain 5 (train, test) splits for the whole dataset
#   - initialize `scores` to an empty list
#   - for each (train, test) split:
#     + run grid-search (ie inner CV loop) on training data and obtain a model
#       fitted to the whole training data with the best hyperparameter.
#     + evaluate the model on the test data
#     + append the resulting score to `scores`
#   - return `scores`
#
# - Grid-search (inner loop):
#   - obtain 5 (train, test) splits for the available data (the training data
#     from the outer loop)
#   - initialize `mean_scores_for_all_params` to an empty list
#   - for each possible hyperparameter value c:
#     + initialize `cv_scores` to an empty list
#     + for each  train, test split:
#       * fit a model on train, using the hyperparameter c
#       * evaluate the model on test
#       * append the resulting score to `cv_scores`
#     + append the mean of `cv_scores` to `mean_scores_for_all_params`
#   - select the hyperparameter with the best mean score
#   - refit the model on the whole available data, using the selected
#     hyperparameter
#   - return this model
#
# Most of this logic is implemented in this module, but some key parts are
# still missing (marked with "TODO"). Your job is to complete the functions
# `cross_validate` and `grid_search` so that the whole nested cross-validation
# can be run.
#
# Some helper routines, `get_train_test_indices`, `fit_and_evaluate`, and
# `expand_param_grid`, are provided to make the task easier. Make sure you read
# their code and understand what they do.
#
# The docstrings of the incomplete functions document precisely what are their
# parameters, and what they should compute and return. Rely on this information
# to write your implementation.
#
# This nested cross-validation procedure is often used, so scikit-learn
# provides all the functionality we are implementing here. To check that our
# implementation is correct, we can therefore compare results with what we
# obtain from scikit-learn. At the end of the file, you will see code that
# loads a scikit-learn dataset, and computes cross-validation scores using our
# `cross_validate` function, then using scikit-learn, and prints both results.
# If you execute this script by running `python nested_cross_validation.py`,
# these two results will be shown and you can check that the code runs and
# produces correct results.

# +
import itertools

import numpy as np
from sklearn.base import clone

# -


# ## Utilities
#
# The 3 functions below are helpers for the main routines `cross_validate` and
# `grid_search`. You should read them but they do not need to be modified.


def get_train_test_indices(n_samples, k=5):
    """Given a total number of samples, return k-fold train, test indices.

    Parameters
    ----------
    n_samples : int
      The total number of samples in the dataset to be split for
      cross-validation.

    k : int, optional (default = 5)
      The number of splits

    Returns
    -------
    splits : list[tuple[np.array[int], np.array[int]]]
      each element of `splits` corresponds to one cross-validation fold and
      contains a pair of arrays (train, test):
      - train: the integer indices of samples in the training set
      - test: the integer indices of samples in the testing set

    """
    indices = np.arange(n_samples)
    test_mask = np.empty(n_samples, dtype=bool)
    splits = []
    start = 0
    for i in range(k):
        n_test = n_samples // k
        if i < n_samples % k:
            n_test += 1
        stop = start + n_test
        test_mask[:] = False
        test_mask[start:stop] = True
        splits.append((indices[np.logical_not(test_mask)], indices[test_mask]))
        start = stop
    return splits


def expand_param_grid(param_grid):
    """
    Construct all possible combinations of parameter values.

    Parameters
    ----------
    `param_grid` : dict[str, list]
      The parameters in the form {"parameter": [list of possible values]}
      For example {"C": [.1, 1.], "penalty": ["l1", "l2"]}

    Returns:
    --------
    `combinations` : list[dict[str, any]]
      a list of all the possible combinations of parameters, for example:
      ```
      [{"C": .1, "penalty": "l1"}, {"C": .1, "penalty": "l2"},
       {"C": 1., "penalty": "l1"}, {"C": 1., "penalty": "l2"}]
      ```

    """
    combinations = []
    for param_values in itertools.product(*param_grid.values()):
        combinations.append(dict(zip(param_grid.keys(), param_values)))
    return combinations


def fit_and_evaluate(model, params, X, y, train_idx, test_idx, score_func):
    """Fit a model on trainig data and compute its score on test data.

    Parameters
    ----------
    model : scikit-learn estimator (will not be modified)
      the estimator to be evaluated

    params : dict[param_name, param_value]

      the hyperparameters to set on (a copy of) the estimator -- an element of
      the list returned by `expand_param_grid`. For example:
      `{"C": .1}`

    X : numpy array of shape (n_samples, n_features)
      the full design matrix

    y : numpy array of shape (n_samples, n_outputs) or (n_samples,)
      the full target vector

    train_idx : sequence of ints
      the indices of training samples (row indices of X)

    test_idx : sequence of ints
      the indicies of testing samples

    score_func : callable
      the function that measures performance on test data, with signature
     `score = score_func(true_y, predicted_y)`.

    Returns
    -------
      The prediction score on test data

    """
    model = clone(model)
    model.set_params(**params)
    model.fit(X[train_idx], y[train_idx])
    predictions = model.predict(X[test_idx])
    score = score_func(y[test_idx], predictions)
    return score


# ## Exercises
#
# The two functions below are incomplete! Complete the body of each function so
# that it behaves as described in the docstring.


def grid_search(model, param_grid, X, y, score_func):
    """Inner loop of a nested cross-validation

    This function estimates the performance of each parameter combination in
    `param_grid` with cross validation. It then selects the best parameters and
    refits a model on the whole data using the selected parameters. The fitted
    model is returned.

    Parameters
    ----------
    model : scikit-learn estimator
      The base estimator, copies of which are trained and evaluated. `model`
      itself is not modified.

    param_grid : dict[str, list]
      grid of possible parameters to try, in the form
      `{"param_name": [list of possible values]}`. For example:
      `{"C": [.01, .1], "penalty": ["l1", "l2"]}`

    X : numpy array of shape (n_samples, n_features)
      the design matrix

    y : numpy array of shape (n_samples, n_outputs) or (n_samples,)
      the target vector

    score_func : callable
      the function computing the score on test data, with signature
      `score = score_func(true_y, predicted_y)`.

    Returns
    -------
    best_model : scikit-learn estimator
      A copy of `model`, fitted on the whole `(X, y)` data, with the
      (estimated) best hyperparameters.

    """
    mean_scores_for_all_params = []
    expanded_param_grid = expand_param_grid(param_grid)
    for params in expanded_param_grid:
        cv_scores = []
        for train_idx, test_idx in get_train_test_indices(len(y)):
            score = fit_and_evaluate(
                model, params, X, y, train_idx, test_idx, score_func
            )
            cv_scores.append(score)
        mean_scores_for_all_params.append(np.mean(cv_scores))
    best_params = expanded_param_grid[np.argmax(mean_scores_for_all_params)]
    best_model = clone(model)
    best_model.set_params(**best_params)
    best_model.fit(X, y)
    return best_model


def cross_validate(model, param_grid, X, y, score_func, k=5):
    """
    Get cross-validation score with an inner CV loop to select hyperparameters.

    Parameters
    ----------
    model : scikit-learn estimator, for example `LogisticRegression()`
      The base model to fit and evaluate. `model` itself is not modified.

    param_grid : dict[str, list]
      grid of possible parameters to try, in the form
      `{"param_name": [list of possible values]}`. For example:
      `{"C": [.01, .1], "penalty": ["l1", "l2"]}`

    X : numpy array of shape (n_samples, n_features)
      the design matrix

    y : numpy array of shape (n_samples, n_outputs) or (n_samples,)
      the target vector

    score_func : callable
      the function computing the score on test data, with signature
      `score = score_func(true_y, predicted_y)`.

    k : int, optional
        the number of splits for the k-fold cross-validation.

    Returns
    -------
    scores : list[float]
       The scores obtained for each of the cross-validation folds

    """
    scores = []
    for train_idx, test_idx in get_train_test_indices(len(y), k=k):
        best_model = grid_search(
            model, param_grid, X[train_idx], y[train_idx], score_func
        )
        predictions = best_model.predict(X[test_idx])
        scores.append(score_func(y[test_idx], predictions))
    return scores


# ## Trying our routines on real data
#
# The code below gets executed when you run this script with `python
# nested_cross_validation`. It computes a cross-validation score with our code,
# and compares it to the results obtained with scikit-learn. Read it to see how
# what we have implemented would be easily done with scikit-learn.
#
# Here we have written this logic ourselves to understand how it works, but in
# practice in real projects we would use the scikit-learn functionality which
# is more flexible, more reliable and faster.
#
# Note: this code will only run once you have completed the exercises!

if __name__ == "__main__":
    from sklearn import datasets, linear_model, model_selection, metrics

    # X, y = datasets.load_wine(return_X_y=True)
    X, y = datasets.load_iris(return_X_y=True)
    idx = np.arange(len(y))
    np.random.default_rng(0).shuffle(idx)
    X, y = X[idx], y[idx]
    model = linear_model.LogisticRegression()
    param_grid = {"C": [0.001, 0.01, 0.1]}
    score_func = metrics.accuracy_score
    my_scores = cross_validate(model, param_grid, X, y, score_func)

    grid_search_model = model_selection.GridSearchCV(
        model, param_grid, scoring="accuracy", cv=model_selection.KFold(5)
    )
    sklearn_scores = model_selection.cross_validate(
        grid_search_model,
        X,
        y,
        scoring="accuracy",
        cv=model_selection.KFold(5),
    )["test_score"]
    print("My scores:")
    print(my_scores)
    print("Scikit-learn scores:")
    print(list(sklearn_scores))
    assert np.allclose(
        my_scores, sklearn_scores
    ), "Results differ from scikit-learn!"
