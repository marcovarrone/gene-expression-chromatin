import numpy as np


def fake_dataset(n_samples, n_landmarks, n_targets):
    X_train = np.random.normal(size=(n_samples, n_landmarks))
    y = np.random.randint(9, size=(n_targets, n_landmarks))
    y_train = X_train.dot(y.T)

    X_val = np.random.normal(size=(n_samples, n_landmarks))
    y_val = X_val.dot(y.T)

    corr = np.identity(n_targets)
    corr[0, 2] = 1
    corr[2, 0] = 1

    return X_train, y_train, corr, X_val, y_val


def fake_dataset_r(n_samples, n_landmarks, n_targets):
    X = np.arange(1, 251).reshape((n_landmarks, n_samples)).T
    Y = np.arange(1, 501).reshape((n_targets, n_samples)).T
    R = np.arange(1, 101).reshape((n_targets, n_targets)).T
    return X, Y, R
