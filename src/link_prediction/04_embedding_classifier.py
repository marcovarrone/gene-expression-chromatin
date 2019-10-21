import argparse
import os

import networkx as nx
import numpy as np
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


def link_centrality(centrality, edges):
    centrality_src = centrality[edges[:, 0]]
    centrality_tgt = centrality[edges[:, 1]]
    centrality_sub = np.abs(centrality_src - centrality_tgt)
    centrality_avg = np.mean(np.vstack((centrality_src, centrality_tgt)), axis=0)
    return centrality_sub, centrality_avg


np.random.seed(42)

model = 'topological'
#model = 'graphsage_n20_10_l10_10_d0.0_r0.01'
#model = 'graphsage_slim_n20_10_l50_8_d0.0_r0.01'

clf_type = 'mlp'
#clf_type = 'lr'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"

if __name__ == '__main__':
    coexpression = np.load(
        '/home/varrone/Prj/gene-expression-chromatin/src/link_prediction/data/GM19238/coexpression_90.npy')
    graph_coexp = nx.from_numpy_array(coexpression)

    edges = np.array(list(graph_coexp.edges))
    n_edges = edges.shape[0]

    non_edges = np.array(list(nx.non_edges(graph_coexp)))
    non_edges = non_edges[np.random.choice(non_edges.shape[0], n_edges, replace=False)]
    n_non_edges = non_edges.shape[0]


    if model == 'topological':
        adj_hic = np.load(
            '/home/varrone/Prj/gene-expression-chromatin/src/link_prediction/data/GM19238/interactions_90.npy')
        graph_hic = nx.from_numpy_array(adj_hic)
        graph_hic = nx.convert_node_labels_to_integers(graph_hic)

        shortest_path_lengths_pos = np.array(list(
            map(lambda e: nx.shortest_path_length(graph_hic, e[0], e[1]) if nx.has_path(graph_hic, e[0],
                                                                                        e[1]) else np.nan,
                edges)))
        shortest_path_lengths_neg = np.array(list(
            map(lambda e: nx.shortest_path_length(graph_hic, e[0], e[1]) if nx.has_path(graph_hic, e[0],
                                                                                        e[1]) else np.nan,
                non_edges)))

        jaccard_index_pos = np.array(list(map(lambda e: e[2], nx.jaccard_coefficient(graph_hic, edges))))
        jaccard_index_neg = np.array(list(map(lambda e: e[2], nx.jaccard_coefficient(graph_hic, non_edges))))

        # Safer alternative (to be fixed)
        # degrees = np.zeros(n_nodes)
        # degrees_dict = dict(graph_hic.degree())
        # degrees[np.array(degrees_dict.keys())] = np.array(degrees_dict.values())

        degrees = np.array(list(dict(graph_hic.degree()).values()))
        degrees_sub_pos, degrees_avg_pos = link_centrality(degrees, edges)
        degrees_sub_neg, degrees_avg_neg = link_centrality(degrees, non_edges)

        betweenness = np.array(list(nx.betweenness_centrality(graph_hic, normalized=True).values()))
        betweenness_sub_pos, betweenness_avg_pos = link_centrality(betweenness, edges)
        betweenness_sub_neg, betweenness_avg_neg = link_centrality(betweenness, non_edges)

        clustering = np.array(list(nx.clustering(graph_hic).values()))
        clustering_sub_pos, clustering_avg_pos = link_centrality(clustering, edges)
        clustering_sub_neg, clustering_avg_neg = link_centrality(clustering, non_edges)

        parameters_pos = np.vstack((shortest_path_lengths_pos, jaccard_index_pos, degrees_sub_pos, degrees_avg_pos,
                                    betweenness_sub_pos, betweenness_avg_pos, clustering_sub_pos, clustering_avg_pos))

        parameters_neg = np.vstack((shortest_path_lengths_neg, jaccard_index_neg, degrees_sub_neg, degrees_avg_neg,
                                    betweenness_sub_neg, betweenness_avg_neg, clustering_sub_neg, clustering_avg_neg))

        #X = np.hstack((parameters_pos, parameters_neg)).T
        #imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        #X = imp.fit_transform(X)
        X = np.hstack((np.vstack((degrees_avg_pos, degrees_avg_pos)), np.vstack((degrees_sub_pos, degrees_sub_neg)))).T

    else:
        embeddings = np.load('embeddings_'+model+'.npy')
        pos_features = np.array(list(map(lambda edge: embeddings[edge[0]]*embeddings[edge[1]], edges)))
        neg_features = np.array(list(map(lambda edge: embeddings[edge[0]]*embeddings[edge[1]], non_edges)))
        X = np.vstack((pos_features, neg_features))
    y = np.hstack((np.ones(n_edges), np.zeros(n_non_edges)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    rocs = []
    accuracies = []
    for i in range(10):
        skf = StratifiedKFold(n_splits=10)
        for train_index, val_index in skf.split(X_train, y_train):
            if clf_type == 'mlp':
                classifier = MLPClassifier(max_iter=500)
            else:
                classifier = LogisticRegression(max_iter=500)
            classifier.fit(X_train[train_index], y_train[train_index])
            y_pred = classifier.predict(X_train[val_index])

            roc = roc_auc_score(y_train[val_index], y_pred)
            print(confusion_matrix(y_train[val_index], y_pred))
            accuracy = classifier.score(X_train[val_index], y_train[val_index])
            rocs.append(roc)
            accuracies.append(accuracy)
            print("Accuracy:", accuracy, "- ROC:", roc)

    #np.save('rocs_'+str(clf_type)+'_'+str(model)+'_degree.npy', rocs)


    print("Mean Accuracy:", np.mean(accuracies), "- Mean ROC:", np.mean(rocs))

