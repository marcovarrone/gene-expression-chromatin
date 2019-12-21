import argparse
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier


def link_centrality(centrality, edges):
    centrality_src = centrality[edges[:, 0]]
    centrality_tgt = centrality[edges[:, 1]]
    centrality_sub = np.abs(centrality_src - centrality_tgt)
    centrality_avg = np.mean(np.vstack((centrality_src, centrality_tgt)), axis=0)
    return centrality_sub, centrality_avg


seed = 42
np.random.seed(seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"


def topological_features(_args, _edges, _non_edges):
    adj_hic = np.load('data/GM19238/interactions/interactions_{}.npy'.format(_args.interactions))
    graph_hic = nx.from_numpy_array(adj_hic)
    graph_hic = nx.convert_node_labels_to_integers(graph_hic)

    degrees = np.array(list(dict(graph_hic.degree()).values()))
    degrees_sub_pos, degrees_avg_pos = link_centrality(degrees, _edges)
    degrees_sub_neg, degrees_avg_neg = link_centrality(degrees, _non_edges)

    betweenness = np.array(list(nx.betweenness_centrality(graph_hic, normalized=True).values()))
    betweenness_sub_pos, betweenness_avg_pos = link_centrality(betweenness, _edges)
    betweenness_sub_neg, betweenness_avg_neg = link_centrality(betweenness, _non_edges)

    clustering = np.array(list(nx.clustering(graph_hic).values()))
    clustering_sub_pos, clustering_avg_pos = link_centrality(clustering, _edges)
    clustering_sub_neg, clustering_avg_neg = link_centrality(clustering, _non_edges)

    node_embs = np.vstack((degrees, betweenness, clustering)).T
    print(node_embs.shape)
    np.save('embeddings/embeddings_chr_{:02d}_topological'.format(_args.chr), node_embs)

    parameters_pos = np.vstack((degrees_sub_pos, degrees_avg_pos,
                                betweenness_sub_pos, betweenness_avg_pos,
                                clustering_sub_pos, clustering_avg_pos))

    parameters_neg = np.vstack((degrees_sub_neg, degrees_avg_neg,
                                betweenness_sub_neg, betweenness_avg_neg,
                                clustering_sub_neg, clustering_avg_neg))

    if _args.edge_features:
        shortest_path_lengths_pos = np.array(list(
            map(lambda e: nx.shortest_path_length(graph_hic, e[0], e[1]) if nx.has_path(graph_hic, e[0],
                                                                                        e[1]) else np.nan,
                _edges)))
        shortest_path_lengths_neg = np.array(list(
            map(lambda e: nx.shortest_path_length(graph_hic, e[0], e[1]) if nx.has_path(graph_hic, e[0],
                                                                                        e[1]) else np.nan,
                _non_edges)))

        jaccard_index_pos = np.array(list(map(lambda e: e[2], nx.jaccard_coefficient(graph_hic, _edges))))
        jaccard_index_neg = np.array(list(map(lambda e: e[2], nx.jaccard_coefficient(graph_hic, _non_edges))))

        parameters_pos = np.vstack((parameters_pos, shortest_path_lengths_pos, jaccard_index_pos))
        parameters_neg = np.vstack((parameters_neg, shortest_path_lengths_neg, jaccard_index_neg))

    X = np.hstack((parameters_pos, parameters_neg)).T
    if _args.edge_features:
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        X = imp.fit_transform(X)
    return X


def evaluate_embedding(X_train, y_train, adj_shape, links, n_iter=1):
    mis_adjacency = np.ones(adj_shape) + 1
    for i in range(n_iter):
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        for train_index, val_index in skf.split(X_train, y_train):
            classifier = MLPClassifier(max_iter=500)
            classifier.fit(X_train[train_index], y_train[train_index])
            y_pred = classifier.predict(X_train[val_index])
            mispredictions = y_train[val_index] - y_pred
            print(f1_score(y_train[val_index], y_pred))
            mis_adjacency[links[val_index, 0], links[val_index, 1]] = mispredictions

    return mis_adjacency


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='GM19238')
    parser.add_argument('--chr', type=int, default=11)
    parser.add_argument('--embedding-pt1', type=str, default='combined_observed_KR')
    parser.add_argument('--embedding-pt2', type=str, default='10000_90_es8')
    #parser.add_argument('--embedding-pt2', type=str, default='10000_90_es8_h8_e3000_lr0.01_do0.3')
    parser.add_argument('--method', type=str, default='pca')
    parser.add_argument('--interactions', type=str)
    parser.add_argument('--edge-features', default=True, action='store_true')
    parser.add_argument('--aggregator', default='hadamard', choices=['hadamard', 'concat'])
    parser.add_argument('--classifier', default='mlp', choices=['mlp', 'lr', 'svm', 'mlp_2'])
    parser.add_argument('--threshold', type=int, default=90)
    args = parser.parse_args()

    coexpression = np.load(
        'data/GM19238/coexpression/coexpression_chr_{:02d}_{}.npy'.format(args.chr, args.threshold))
    graph_coexp = nx.from_numpy_array(coexpression)

    edges = np.array(list(graph_coexp.edges))
    n_edges = edges.shape[0]

    non_edges = np.array(list(nx.non_edges(graph_coexp)))
    non_edges = non_edges[np.random.choice(non_edges.shape[0], n_edges, replace=False)]
    n_non_edges = non_edges.shape[0]

    if args.embedding_pt1 == 'topological':
        X = topological_features(args, edges, non_edges)
    else:
        embeddings = np.load(
            './embeddings/{}/{}_{}_{}_{}.npy'.format(args.method, args.embedding_pt1, args.chr, args.chr,
                                                     args.embedding_pt2))
        if args.aggregator == 'hadamard':
            pos_features = np.array(list(map(lambda edge: embeddings[edge[0]] * embeddings[edge[1]], edges)))
            neg_features = np.array(list(map(lambda edge: embeddings[edge[0]] * embeddings[edge[1]], non_edges)))
        else:
            pos_features = np.array(list(map(lambda edge: np.hstack(embeddings[edge[0]], embeddings[edge[1]]), edges)))
            neg_features = np.array(
                list(map(lambda edge: np.hstack(embeddings[edge[0]], embeddings[edge[1]]), non_edges)))
        X = np.vstack((pos_features, neg_features))
    y = np.hstack((np.ones(n_edges), np.zeros(n_non_edges)))
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    mis_adj = evaluate_embedding(X, y, coexpression.shape, np.concatenate((edges, non_edges)))
    plt.imshow(mis_adj, cmap='gist_ncar')
    plt.show()
