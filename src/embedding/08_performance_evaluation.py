import argparse

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from mlp import MLP

parser = argparse.ArgumentParser()

parser.add_argument('-t', '--training-set', type=str)
parser.add_argument('-v', '--validation-set', type=str)
parser.add_argument('-l', '--landmarks', type=str, default='l1000')
parser.add_argument('-r', '--random-landmarks', type=int)
parser.add_argument('-m', '--model', type=str, default='lr')

parser.add_argument('--dataset', type=str, default='GSE92743')

args = parser.parse_args()

train = np.load('data/' + str(args.dataset) + '/' + str(args.training_set) + '.npy').T
print('Normalizing')
scaler = StandardScaler()
train = scaler.fit_transform(train)


def total_mae(Y_true, Y_pred):
    return np.mean(np.sum(np.abs(Y_true - Y_pred), axis=0) / Y_true.shape[0])


if args.random_landmarks:
    landmarks = np.random.choice(12321, args.random_landmarks, replace=False)
else:
    landmarks = np.load('landmarks/' + str(args.dataset) + '/' + str(args.landmarks) + '.npy')

# ToDo: avoid hard coding
targets = np.arange(12320)
targets = np.setdiff1d(targets, landmarks)

X_train, Y_train = train[:, landmarks], train[:, targets]

if args.validation_set:
    scaler = StandardScaler()
    valid = np.load('data/' + str(args.dataset) + '/' + str(args.validation_set) + '.npy').T
    valid = scaler.fit_transform(valid)
    X_valid, Y_valid = valid[:, landmarks], valid[:, targets]

model = None
if args.model == 'lr':
    model = LinearRegression(n_jobs=10)
    model.fit(X_train, Y_train)
elif args.model == 'mlp':
    if args.validation_set:
        validation_data = (X_valid, Y_valid)
    else:
        validation_data = None
    model = MLP(X_train.shape[1], Y_train.shape[1], 1, 4000, learning_rate=0.0001)
    model.fit(X_train, Y_train, batch_size=128, epochs=100, validation_data=validation_data)

if args.validation_set:
    Y_pred = model.predict(X_valid)
    print(total_mae(Y_valid, Y_pred))
