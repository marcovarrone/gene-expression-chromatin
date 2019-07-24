import argparse

import numpy as np
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()

parser.add_argument('-t', '--training-set', type=str)
parser.add_argument('-v', '--validation-set', type=str)
parser.add_argument('-l', '--landmarks', type=str, default='l1000')

parser.add_argument('--dataset', type=str, default='GSE92743')

args = parser.parse_args()

train = np.load('data/' + str(args.dataset) + '/' + str(args.training_set) + '.npy')


def total_mae(Y_true, Y_pred):
    return np.mean(np.sum(np.abs(Y_true - Y_pred), axis=0) / Y_true.shape[0])


landmarks = np.load('landmarks/' + str(args.dataset) + '/' + str(args.landmarks) + '.npy')
targets = np.arange(train.shape[1])
targets = np.setdiff1d(targets, landmarks)

X_train, Y_train = train[:, landmarks], train[:, targets]

if args.validation_set:
    valid = np.load('data/' + str(args.dataset) + '/' + str(args.validation_set) + '.npy')
    X_valid, Y_valid = valid[:, landmarks], valid[:, targets]

model = LinearRegression(n_jobs=10)
model.fit(X_train, Y_train)

if args.validation_set:
    Y_pred = model.predict(X_valid)
    print(total_mae(Y_valid, Y_pred))
