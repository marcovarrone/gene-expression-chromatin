import argparse

import networkx as nx
import numpy as np
import wandb
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neural_network import MLPClassifier

wandb.init(project="prova")
wandb.config.classifier = 'MLP'

parser = argparse.ArgumentParser()
parser.add_argument('--walk-length', type=int)
parser.add_argument('--num-walks', type=int)
parser.add_argument('--p', type=float)
parser.add_argument('--q', type=float)
parser.add_argument('--window', type=int)
parser.add_argument('--dimensions', type=int)
args = parser.parse_args()

wandb.config.update(args)

np.random.seed(42)

if __name__ == '__main__':
    coexpression = np.load(
        '/home/varrone/Prj/gene-expression-chromatin/src/link_prediction/data/GM19238/coexpression_90.npy')
    graph_coexp = nx.from_numpy_array(coexpression)

    adj = np.load('/home/varrone/Prj/gene-expression-chromatin/src/link_prediction/data/GM19238/interactions_80.npy')
    graph = nx.from_numpy_array(adj)

    node2vec = Node2Vec(graph, dimensions=args.dimensions, walk_length=args.walk_length, num_walks=args.num_walks,
                        p=args.p, q=args.q, workers=4)

    model = node2vec.fit(window=args.window, min_count=0)

    edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

    edges = np.array([tuple(map(str, e)) for e in graph_coexp.edges])
    n_edges = edges.shape[0]
    '''train_pos = np.arange(n_edges)
    train_pos, val_pos = edge_test_sample(train_pos, 0.1)
    train_pos, test_pos = edge_test_sample(train_pos, 0.1)'''

    non_edges = np.array([tuple(map(str, e)) for e in nx.non_edges(graph_coexp)])
    non_edges = non_edges[np.random.choice(non_edges.shape[0], n_edges, replace=False)]
    n_non_edges = non_edges.shape[0]
    '''train_neg = np.arange(n_non_edges)
    train_neg, val_neg = edge_test_sample(train_neg, 0.1)
    train_neg, test_neg = edge_test_sample(train_neg, 0.1)'''
    pos_features = np.array(list(map(lambda edge: edges_embs[(edge[0], edge[1])], edges)))
    neg_features = np.array(list(map(lambda edge: edges_embs[(edge[0], edge[1])], non_edges)))
    X = np.vstack((pos_features, neg_features))
    y = np.hstack((np.ones(n_edges), np.zeros(n_non_edges)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    skf = StratifiedKFold(n_splits=5)
    rocs = []
    accuracies = []
    for train_index, val_index in skf.split(X_train, y_train):
        #classifier = LogisticRegression()
        classifier = MLPClassifier(max_iter=500)
        classifier.fit(X_train[train_index], y_train[train_index])
        y_pred = classifier.predict(X_train[val_index])

        roc = roc_auc_score(y_train[val_index], y_pred)
        accuracy = classifier.score(X_train[val_index], y_train[val_index])
        rocs.append(roc)
        accuracies.append(accuracy)
        print("Accuracy:", accuracy, "- ROC:", roc)

    wandb.run.summary["roc_mean"] = np.mean(rocs)
    wandb.run.summary["roc_std"] = np.std(rocs)
    wandb.run.summary["acc_mean"] = np.mean(accuracies)
    wandb.run.summary["acc_std"] = np.std(accuracies)
    print("Mean Accuracy:", np.mean(accuracies), "- Mean ROC:", np.mean(rocs))

