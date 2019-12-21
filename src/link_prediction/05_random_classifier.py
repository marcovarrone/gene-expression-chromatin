import argparse
import os

import networkx as nx
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def link_centrality(centrality, edges):
    centrality_src = centrality[edges[:, 0]]
    centrality_tgt = centrality[edges[:, 1]]
    centrality_sub = np.abs(centrality_src - centrality_tgt)
    centrality_avg = np.mean(np.vstack((centrality_src, centrality_tgt)), axis=0)
    return centrality_sub, centrality_avg


np.random.seed(42)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MCF7')
    parser.add_argument('--chr', type=int)
    parser.add_argument('--threshold', type=float, default=0.28)
    args = parser.parse_args()

    coexpression = np.load(
        'data/{}/coexpression/coexpression_chr_{:02d}_{:02d}_{}.npy'.format(args.dataset, args.chr, args.chr,
                                                                            args.threshold))
    graph_coexp = nx.from_numpy_array(coexpression)

    edges = np.array(list(graph_coexp.edges))
    n_edges = edges.shape[0]

    non_edges = np.array(list(nx.non_edges(graph_coexp)))
    non_edges = non_edges[np.random.choice(non_edges.shape[0], n_edges, replace=False)]
    n_non_edges = non_edges.shape[0]

    X = np.vstack((edges, non_edges))

    y = np.hstack((np.ones(n_edges), np.zeros(n_non_edges)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    rocs = []
    accuracies = []
    f1s = []
    precisions = []
    recalls = []
    for i in range(10):
        skf = StratifiedKFold(n_splits=10)
        for train_index, val_index in skf.split(X_train, y_train):

            y_pred = np.random.choice([0, 1], y_train[val_index].shape[0])

            roc = roc_auc_score(y_train[val_index], y_pred)
            accuracy = accuracy_score(y_train[val_index], y_pred)
            f1 = f1_score(y_train[val_index], y_pred)
            precision = precision_score(y_train[val_index], y_pred)
            recall = recall_score(y_train[val_index], y_pred)
            rocs.append(roc)
            accuracies.append(accuracy)
            f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            print("Accuracy:", accuracy, "- ROC:", roc, "- F1:", f1, "- Precision:", precision, "- Recall:", recall)

    if not os.path.exists('results/{}/chr_{:02d}'.format(args.dataset, args.chr)):
        os.makedirs('results/{}/chr_{:02d}'.format(args.dataset, args.chr))

    with open('results/{}/chr_{:02d}/random.pkl'.format(args.dataset, args.chr), 'wb') as file_save:
        pickle.dump({'acc': accuracies, 'roc': rocs, 'f1': f1s, 'precision': precisions, 'recall': recalls}, file_save)

    print("Mean Accuracy:", np.mean(accuracies), "- Mean ROC:", np.mean(rocs), "- Mean F1:", np.mean(f1s),
          "- Mean Precision:", np.mean(precisions), "- Mean Recall", np.mean(recalls))
