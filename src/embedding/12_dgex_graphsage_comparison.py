import argparse
import configparser
import math
import os

import numpy as np
import scipy.sparse as sps
import torch
import wandb
from sklearn.preprocessing import StandardScaler

from models.graphsage_dgl import GraphSAGEWrapper
from models.mlp_pt import MLP

config = configparser.ConfigParser()
config.read('/home/varrone/config.ini')

np.random.seed(42)
torch.manual_seed(42)


def normalize(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def total_mae(Y_true, Y_pred):
    return np.mean(np.sum(np.abs(Y_true - Y_pred), axis=0) / Y_true.shape[0])


'''def graphsage_evaluation(X, Y, train_mask, contact_matrix, n_hidden, n_layers, n_epochs, lr, input_dropout, dropout,
                         activation):
    kf = KFold(n_splits=5)
    maes = []
    test_mask = np.random.choice(train_mask, int(train_mask.shape[0] * 0.2), replace=False)
    train_mask = np.setdiff1d(train_mask, test_mask)
    # print(X[train_mask].shape[0] * X[train_mask].shape[1], Y[train_mask].shape[0] * Y[train_mask].shape[1])
    for train_index, val_index in kf.split(train_mask):
        train_mask_cv = train_mask[train_index]
        val_mask_cv = train_mask[val_index]
        model = GraphSAGEWrapper(X.shape[1], Y.shape[1], contact_matrix, n_hidden=n_hidden, n_layers=n_layers,
                                 input_dropout=input_dropout, dropout=dropout, activation=activation)
        # print(X[train_mask_cv].shape)
        model.fit(X, Y, train_mask_cv, val_mask_cv, n_epochs=n_epochs, lr=lr)
        mae = model.evaluate(X, Y, val_mask_cv)
        print("Validation MAE {:.4f}".format(mae))
        maes.append(mae)
    print(np.mean(maes))
    print(np.std(maes))

    model = GraphSAGEWrapper(X.shape[1], Y.shape[1], contact_matrix, n_hidden=n_hidden, n_layers=n_layers,
                             input_dropout=input_dropout, dropout=dropout, activation=activation)
    model.fit(X, Y, train_mask, n_epochs=n_epochs, lr=lr)
    mae = model.evaluate(X, Y, test_mask)
    print("Test MAE {:.4f}".format(mae))


def mlp_evaluation(X, Y, train_mask):
    kf = KFold(n_splits=5)
    maes = []
    test_mask = np.random.choice(train_mask, int(train_mask.shape[0] * 0.2), replace=False)
    train_mask = np.setdiff1d(train_mask, test_mask)
    for train_index, val_index in kf.split(train_mask):
        train_mask_cv = train_mask[train_index]
        val_mask_cv = train_mask[val_index]
        model = MLP(X.shape[1], Y.shape[1], 2000, 1, activation=torch.nn.Tanh)
        # print(summary(model.model, X[train_mask_cv].shape, device="cpu"))
        model.fit(X[train_mask_cv], Y[train_mask_cv], validation_data=(X[val_mask_cv], Y[val_mask_cv]), n_epochs=50,
                  lr=0.00001)
        mae = model.evaluate(X[val_mask_cv], Y[val_mask_cv])
        print("Validation MAE {:.4f}".format(mae))
        maes.append(mae)
    print(np.mean(maes))
    print(np.std(maes))

    dataloader_train = DatasetGM(X[train_mask], Y[train_mask])
    model = MLP(X.shape[1], Y.shape[1], 100, 1)
    model.fit(dataloader_train, n_epochs=800, lr=0.001)
    mae = model.evaluate(X[test_mask], Y[test_mask])
    print("Test MAE {:.4f}".format(mae))'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='GM19238')
    parser.add_argument('-tr', '--training-set', type=str, default='X_train_59_0_normalized')
    parser.add_argument('-v', '--validation-size', type=float)
    parser.add_argument('-te', '--test-size', type=float)
    parser.add_argument('--n-jobs', type=int, default=10)
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('-m', '--model', default='mlp')
    parser.add_argument('--hidden-size', type=int)
    parser.add_argument('--n-layers', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('-lr', '--learning-rate', type=float)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--in-dropout', type=float, default=0)
    parser.add_argument('--aggregation-type', type=str, default="mean")
    parser.add_argument('--activation', type=str, default="tanh", choices=['tanh', 'relu'])

    parser.add_argument('--wandb', default=False, action='store_true')

    args = parser.parse_args()

    torch.set_num_threads(args.n_jobs)

    os.environ["OMP_NUM_THREADS"] = str(args.n_jobs)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.n_jobs)
    os.environ["MKL_NUM_THREADS"] = str(args.n_jobs)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.n_jobs)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.n_jobs)

    if not args.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if args.wandb:
        wandb.init(project="graphsage-mlp-comparison")
    else:
        wandb = None

    if args.activation == 'tanh':
        activation = torch.tanh
    else:
        activation = torch.relu

    dataset = np.load('data/' + str(args.dataset) + '/' + str(args.training_set) + '.npy')
    n_genes = dataset.shape[0]
    n_samples = dataset.shape[1]
    genes_Y = np.random.choice(np.arange(n_genes), math.ceil(n_genes * args.test_size), replace=False)
    genes_X = np.setdiff1d(np.arange(n_genes), genes_Y)

    samples_Y = np.random.choice(np.arange(n_samples), math.ceil(n_samples * args.test_size), replace=False)
    samples_X = np.setdiff1d(np.arange(n_samples), samples_Y)

    if args.dataset == 'GSE92743':
        adjacency = sps.load_npz(config['GENEMANIA'][args.dataset]).todense()
        adjacency /= np.max(adjacency)
        if not args.self_loop:
            np.fill_diagonal(adjacency, 1)
    else:
        #adjacency = np.load(
        #    '/home/varrone/Prj/gene-expression-chromatin/src/preprocessing/GM19238/GSE63525_GM12878_insitu_primary_10kb_contact_matrix.npy')
        #adjacency = np.log2(adjacency + 1)
        adjacency = np.eye(n_genes, n_genes)

    dataset = StandardScaler().fit_transform(dataset)
    X_gs, Y_gs = dataset[:, samples_X], dataset[:, samples_Y]
    train_mask_gs, test_mask_gs = genes_X, genes_Y

    # graphsage_evaluation(X_gs, Y_gs, genes_X, adjacency, 2000, 1, 400, 0.001, 0.0, 0, torch.tanh)

    dataset_mlp = dataset.T
    X_mlp, Y_mlp = dataset_mlp[:, genes_X], dataset_mlp[:, genes_Y]
    train_mask_mlp, test_mask_mlp = samples_X, samples_Y

    # mlp_evaluation(X_mlp, Y_mlp, train_mask=train_mask_mlp)

    if args.model == 'graphsage':
        val_mask = None
        if args.validation_size:
            val_mask = np.random.choice(train_mask_gs, int(train_mask_gs.shape[0] * args.validation_size),
                                        replace=False)
            train_mask_gs = np.setdiff1d(train_mask_gs, val_mask)

        model = GraphSAGEWrapper(X_gs.shape[1], Y_gs.shape[1], adjacency, n_hidden=args.hidden_size,
                                 n_layers=args.n_layers, activation=activation, wandb=wandb, dropout=args.dropout,
                                 input_dropout=args.in_dropout, aggregator_type=args.aggregation_type)
        model.fit(X_gs, Y_gs, train_mask_gs, n_epochs=args.epochs, lr=args.learning_rate, val_mask=val_mask)
        if args.test:
            mae = model.test(X_gs, Y_gs, test_mask_gs)
            print(mae)
    elif args.model == 'mlp':
        val_mask = np.random.choice(train_mask_mlp, int(train_mask_gs.shape[0] * args.validation_size), replace=False)
        train_mask_mlp = np.setdiff1d(train_mask_mlp, val_mask)

        model = MLP(X_mlp[train_mask_mlp].shape[1], Y_mlp[train_mask_mlp].shape[1], hidden_size=args.hidden_size,
                    n_layers=args.n_layers, input_dropout=args.in_dropout, dropout=args.dropout, wandb=wandb,
                    activation=activation)
        model.fit(X_mlp[train_mask_mlp], Y_mlp[train_mask_mlp], validation_data=(X_mlp[val_mask], Y_mlp[val_mask]),
                  n_epochs=args.epochs, lr=args.learning_rate)
        if args.test:
            mae = model.evaluate(X_mlp[test_mask_mlp], Y_mlp[test_mask_mlp])
            print(mae)
