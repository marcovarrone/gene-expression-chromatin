import argparse
import os

from models.mlp import MLP

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=False, action='store_true')
parser.add_argument('-t', '--training-set', type=str, required=True)
parser.add_argument('-v', '--validation-set', type=str, required=True)
parser.add_argument('-d', '--dataset', type=str, required=True)
parser.add_argument('--n-jobs', type=int, default=15)
parser.add_argument('--n-samplings', type=int, default=10)
parser.add_argument('--random-seed', type=int, default=42)
parser.add_argument('--save-fig-errors', default=False, action='store_true')
parser.add_argument('--save-fig-params', default=False, action='store_true')
parser.add_argument('--save-errors', default=False, action='store_true')
parser.add_argument('--l1000-line', default=False, action='store_true')
parser.add_argument('--w-evaluation', default=False, action='store_true')
parser.add_argument('-m', '--model', type=str, default='lr', choices=['lr', 'mlp'])
parser.add_argument('--force-run', default=False, action='store_true')
parser.add_argument('--best-n-landmarks', default=False, action='store_true')

parser.add_argument('--hidden-size', type=int, default=2000)

parser.add_argument('--min-low', type=int, default=1)
parser.add_argument('--max-low', type=int, default=2000)
parser.add_argument('--step-low', type=int, default=100)
parser.add_argument('--min-high', type=int, default=2000)
parser.add_argument('--max-high', type=int, default=12000)
parser.add_argument('--step-high', type=int, default=1000)

args = parser.parse_args()

if not args.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ["OMP_NUM_THREADS"] = str(args.n_jobs)
os.environ["OPENBLAS_NUM_THREADS"] = str(args.n_jobs)
os.environ["MKL_NUM_THREADS"] = str(args.n_jobs)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.n_jobs)
os.environ["NUMEXPR_NUM_THREADS"] = str(args.n_jobs)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pandas as pd
import seaborn as sns

performance_evaluation = __import__('08_performance_evaluation')

np.random.seed(args.random_seed)
tf.set_random_seed(args.random_seed)


def total_mae(Y_true, Y_pred):
    return np.mean(np.sum(np.abs(Y_true - Y_pred), axis=0) / Y_true.shape[0])


def normalize(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)


train = np.load('data/' + str(args.dataset) + '/' + str(args.training_set) + '.npy').T
train = normalize(train)
valid = np.load('data/' + str(args.dataset) + '/' + str(args.validation_set) + '.npy').T
valid = normalize(valid)

filename = 'lm_error_plot_' + str(args.training_set) + '_' + str(args.model)
if args.model == 'mlp':
    filename += '_' + str(args.hidden_size)
filename += '_s' + str(args.n_samplings) + '_' + str(args.min_low) + '_' + str(args.max_high)

print(filename)
n_params = []

landmarks_list = np.array([], dtype=np.int)
if args.step_low:
    landmarks_low = np.arange(args.min_low, args.max_low, args.step_low)
    landmarks_list = np.hstack((landmarks_list, landmarks_low))

if args.step_high:
    landmarks_high = np.arange(args.min_high, args.max_high, args.step_high)
    landmarks_list = np.hstack((landmarks_list, landmarks_high))

n_landmarks_l1000 = len(np.load('landmarks/' + str(args.dataset) + '/l1000.npy'))

if __name__ == '__main__':
    if os.path.isfile('lm_error_plots/' + str(args.dataset) + '/' + str(filename) + '.pkl') and not args.force_run:
        print("The error values have been already computed. Loading from file")
        df = pd.read_pickle('lm_error_plots/' + str(args.dataset) + '/' + str(filename) + '.pkl')
    else:
        maes = []
        landmarks_points = []

        for n_landmarks in landmarks_list:
            for i in range(args.n_samplings):
                if i == 0:
                    print("N. landmarks", n_landmarks)
                print(i + 1, "/", args.n_samplings)
                landmarks = np.random.choice(train.shape[1], n_landmarks, replace=False)
                targets = np.arange(train.shape[1])
                targets = np.setdiff1d(targets, landmarks)
                X_train, Y_train = train[:, landmarks], train[:, targets]
                X_valid, Y_valid = valid[:, landmarks], valid[:, targets]
                # X_test, Y_test = test[:, landmarks], test[:, targets]
                X_test = X_valid
                Y_test = Y_valid

                if args.model == 'lr':
                    model = LinearRegression(n_jobs=10)
                    model.fit(X_train, Y_train)
                    Y_pred = model.predict(X_test)
                    if i == 0:
                        n_params.append(X_train.shape[1] * Y_train.shape[1])
                else:
                    model = MLP(X_train.shape[1], Y_train.shape[1], 1, args.hidden_size, patience=5)
                    model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), batch_size=64)
                    Y_pred = model.predict(X_test, batch_size=64)

                    if i == 0:
                        n_params.append(model.count_params())
                mae = total_mae(Y_test, Y_pred)
                print(mae)
                landmarks_points.append(n_landmarks)
                maes.append(mae)

        print(maes)
        df = pd.DataFrame(data={'n_landmarks': landmarks_points, 'errors': maes})

        if args.save_errors:
            df.to_pickle('lm_error_plots/' + str(args.dataset) + '/' + str(filename) + '.pkl')

    if args.save_fig_errors:
        ax = sns.lineplot(x="n_landmarks", y="errors", data=df)
        ax.set(xlabel='N. landmarks', ylabel='Mean Average Error')
        filename_fig = filename
        if args.l1000_line:
            filename_fig += '_line'
            df_mean = df.groupby(['n_landmarks']).mean()
            l1000_y = np.interp(n_landmarks_l1000, df_mean.index, df_mean['errors'])
            plt.plot([n_landmarks_l1000, n_landmarks_l1000], [min(df_mean['errors']), l1000_y], 'r--')
            plt.plot([min(df_mean.index), n_landmarks_l1000], [l1000_y, l1000_y], 'r--')
        if args.w_evaluation:
            filename_fig += '_eval'
            mae = performance_evaluation.evaluate(train, valid, model=args.model, hidden_size=args.hidden_size, dataset=args.dataset, landmarks='l1000')
            print("MAE for L1000 landmarks is", mae)
            plt.plot(n_landmarks_l1000, mae, 'ob')
        plt.savefig('lm_error_plots/' + str(args.dataset) + '/' + str(filename_fig) + '.png')
        plt.clf()

    if args.save_fig_params:
        df = pd.DataFrame(data={'n_landmarks': landmarks_list, 'n_parameters': n_params})
        ax = sns.lineplot(x="n_landmarks", y="n_parameters", data=df)
        ax.set(xlabel='N. landmarks', ylabel='N. parameters')
        plt.savefig('lm_error_plots/' + str(args.dataset) + '/' + str(filename) + '_params.png')
        plt.show()

    if args.best_n_landmarks:
        df_mean = df.groupby(['n_landmarks'], as_index=False).mean()
        min_idx = df_mean['errors'].idxmin()
        print(df_mean['n_landmarks'][min_idx])