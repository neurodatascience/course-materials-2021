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
#   - obtain 3 (train, test) splits for the available data (the training data
#     from the outer loop)
#   - initialize `hyperparameter_scores` to an empty list
#   - for each possible hyperparameter value C:
#     + initialize `cv_scores` to an empty list
#     + for each  train, test split:
#       * fit a model on train, using the hyperparameter C
#       * evaluate the model on test
#       * append the resulting score to `cv_scores`
#     + append the mean of `cv_scores` to `hyperparameter_scores`
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
# Some helper routines, `get_train_test_indices` and `fit_and_evaluate`, are
# provided to make the task easier. Make sure you read their code and
# understand what they do.
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
import numpy as np
from sklearn.base import clone

# -


# ## Utilities
#
# The 2 functions below are helpers for the main routines `cross_validate` and
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


def fit_and_evaluate(model, C, X, y, train_idx, test_idx, score_func):
    """Fit a model on trainig data and compute its score on test data.

    Parameters
    ----------
    model : scikit-learn estimator (will not be modified)
      the estimator to be evaluated

    C : float
     The value for the regularization hyperparameter C to use when fitting the
     model.

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
    model.set_params(C=C)
    model.fit(X[train_idx], y[train_idx])
    predictions = model.predict(X[test_idx])
    score = score_func(y[test_idx], predictions)
    print(
        f"    Inner CV loop: fit and evaluate one model; score = {score:.2f}"
    )
    return score


# ## Exercises
#
# The two functions below are incomplete! Complete the body of each function so
# that it behaves as described in the docstring.


def grid_search(model, hyperparam_grid, X, y, score_func):
    """Inner loop of a nested cross-validation

    This function estimates the performance of each hyperparameter in
    `hyperparam_grid` with cross validation. It then selects the best
    hyperparameter and refits a model on the whole data using the selected
    hyperparameter. The fitted model is returned.

    Parameters
    ----------
    model : scikit-learn estimator
      The base estimator, copies of which are trained and evaluated. `model`
      itself is not modified.

    hyperparam_grid : list[float]
      list of possible values for the hyperparameter C.

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
      (estimated) best hyperparameter.

    """
    hyperparameter_scores = []
    for C in hyperparam_grid:
        print(f"  Grid search: evaluate hyperparameter C = {C}")
        # **TODO** : run 3-fold cross-validation loop, using this particular
        # hyperparameter C. Compute the mean of scores accross cross-validation
        # folds and append it to `hyperparameter_scores`.
        mean_score = "TODO"
        hyperparameter_scores.append(mean_score)
    # **TODO**: select the best hyperparameter according to the CV scores,
    # refit the model on the whole data using this hyperparameter, and return
    # the fitted model. Use `model.set_params` to set the hyperparameter
    best_C = "TODO"
    print(f"  ** Grid search: keep best hyperparameter C = {best_C} **")
    # `clone` is to work with a copy of `model` instead of modifying the
    # argument itself.
    best_model = clone(model)
    # ...
    return best_model


def cross_validate(model, hyperparam_grid, X, y, score_func, k=5):
    """
    Get cross-validation score with an inner CV loop to select hyperparameters.

    Parameters
    ----------
    model : scikit-learn estimator, for example `LogisticRegression()`
      The base model to fit and evaluate. `model` itself is not modified.

    hyperparam_grid : list[float]
      list of possible values for the hyperparameter C.

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
    for i, (train_idx, test_idx) in enumerate(
        get_train_test_indices(len(y), k=k)
    ):
        print(f"\nOuter CV loop: fold {i}")
        # **TODO**: complete the cross-validation loop. For each train, test
        # split, use `grid_search` to run an inner cross-validation loop on the
        # train data, then evaluate the resulting model on test data and store
        # the resulting score.
        score = "TODO"
        print(f"Outer CV loop: finished fold {i}, score: {score:.2f}")
        scores.append(score)
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
    np.random.RandomState(0).shuffle(idx)
    # this is now the recommended way of doing this, but only works with recent
    # versions of numpy:
    # np.random.default_rng(0).shuffle(idx)
    X, y = X[idx], y[idx]
    model = linear_model.LogisticRegression()
    hyperparam_grid = [0.0001, 0.001, 0.01, 0.1]
    score_func = metrics.accuracy_score
    my_scores = cross_validate(model, hyperparam_grid, X, y, score_func)

    grid_search_model = model_selection.GridSearchCV(
        model,
        {"C": hyperparam_grid},
        scoring="accuracy",
        cv=model_selection.KFold(3),
    )
    sklearn_scores = model_selection.cross_validate(
        grid_search_model,
        X,
        y,
        scoring="accuracy",
        cv=model_selection.KFold(5),
    )["test_score"]
    print("\n\nMy scores:")
    print(my_scores)
    print("Scikit-learn scores:")
    print(list(sklearn_scores))
    assert np.allclose(
        my_scores, sklearn_scores
    ), "Results differ from scikit-learn!"

# ## Questions
#
# - When running the cross-validation procedure we have just implemented, how
#   many models did we fit in total?
# - There are 150 samples in the iris dataset. For this dataset, what is the
#   size of the 5 test sets in the outer loop? of each of the 3 validation sets
#   in the grid-search (inner loop)?
#
# ## Additional exercise (optional)
#
# Have you noticed the hyperparameter grid was specified slightly differently
# for the scikit-learn `GridSearchCV`? we passed a dictionary:
# `{"C": [0.0001, 0.001, 0.01, 0.1 ]}`.
#
# This is because with `GridSearchCV` we can specify values for several
# hyperparameters, for example:
# `{"C": [0.0001, 0.001, 0.01, 0.1], "penalty": ["l1", "l2"]}`,
# and all combinations of these will be tried.
#
# Modify this module so that we can specify such a hyperparameter grid, rather
# than only a list of values for a specific hyperparameter named "C". Hint:
# check the documentation for the `set_params` function of scikit-learn
# estimators. You may want to use the python dict unpacking syntax, for example
# `model.set_params(**hyperparams)`. You can also use `itertools.product` from
# the python standard library to easily build all the combinations of
# hyperparameters.
