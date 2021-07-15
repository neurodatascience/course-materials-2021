# # Performing nested cross-validation
import itertools

import numpy as np
from sklearn.base import clone


def train_test_indices(n_samples, k=5):
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
    combinations = []
    for param_values in itertools.product(*param_grid.values()):
        combinations.append(dict(zip(param_grid.keys(), param_values)))
    return combinations


def fit_and_evaluate(model, params, X, y, train_idx, test_idx, score_func):
    model = clone(model)
    model.set_params(**params)
    model.fit(X[train_idx], y[train_idx])
    predictions = model.predict(X[test_idx])
    score = score_func(y[test_idx], predictions)
    return score


def grid_search(model, param_grid, X, y, score_func):
    mean_scores_for_all_params = []
    expanded_param_grid = expand_param_grid(param_grid)
    for params in expanded_param_grid:
        cv_scores = []
        for train_idx, test_idx in train_test_indices(len(y)):
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


def cross_validate(model, param_grid, X, y, score_func):
    scores = []
    for train_idx, test_idx in train_test_indices(len(y)):
        best_model = grid_search(
            model, param_grid, X[train_idx], y[train_idx], score_func
        )
        predictions = best_model.predict(X[test_idx])
        scores.append(score_func(y[test_idx], predictions))
    return scores


if __name__ == "__main__":
    from sklearn import datasets, linear_model, model_selection, metrics

    X, y = datasets.load_wine(return_X_y=True)
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
    print(my_scores)
    print(list(sklearn_scores))
