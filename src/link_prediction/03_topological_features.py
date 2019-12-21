import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neural_network import MLPClassifier


def edge_test_sample(edges, test_size=0.1):
    n_edges = len(edges)
    test_edges = np.random.choice(edges, int(n_edges * test_size), replace=False)
    train_edges = np.setdiff1d(edges, test_edges)
    return train_edges, test_edges


def link_centrality(centrality, edges):
    centrality_src = centrality[edges[:, 0]]
    centrality_tgt = centrality[edges[:, 1]]
    centrality_sub = np.abs(centrality_src - centrality_tgt)
    centrality_avg = np.mean(np.vstack((centrality_src, centrality_tgt)), axis=0)
    return centrality_sub, centrality_avg


np.random.seed(42)
if __name__ == '__main__':
    adj_hic = np.load(
        '/home/varrone/Prj/gene-expression-chromatin/src/link_prediction/data/MCF7/interactions/interactions_primary_oe_NONE_2_2_10000_40000_sum_0.0.npy')
    graph_hic = nx.from_numpy_array(adj_hic)
    n_nodes = nx.number_of_nodes(graph_hic)

    '''adj_coexp = np.load(
        '/home/varrone/Prj/gene-expression-chromatin/src/link_prediction/data/GM19238/coexpression_90.npy')
    graph_coexp = nx.from_numpy_array(adj_coexp)

    edges = np.array(list(graph_coexp.edges))
    n_edges = edges.shape[0]

    non_edges = np.array(list(nx.non_edges(graph_coexp)))
    non_edges = non_edges[np.random.choice(non_edges.shape[0], n_edges, replace=False)]
    n_non_edges = non_edges.shape[0]

    shortest_path_lengths_pos = np.array(list(
        map(lambda e: nx.shortest_path_length(graph_hic, e[0], e[1]) if nx.has_path(graph_hic, e[0], e[1]) else np.nan,
            edges)))
    shortest_path_lengths_neg = np.array(list(
        map(lambda e: nx.shortest_path_length(graph_hic, e[0], e[1]) if nx.has_path(graph_hic, e[0], e[1]) else np.nan,
            non_edges)))

    jaccard_index_pos = np.array(list(map(lambda e: e[2], nx.jaccard_coefficient(graph_hic, edges))))
    jaccard_index_neg = np.array(list(map(lambda e: e[2], nx.jaccard_coefficient(graph_hic, non_edges))))'''

    # Safer alternative (to be fixed)
    # degrees = np.zeros(n_nodes)
    # degrees_dict = dict(graph_hic.degree())
    # degrees[np.array(degrees_dict.keys())] = np.array(degrees_dict.values())

    degrees = np.array(list(dict(graph_hic.degree()).values()))
    #degrees_sub_pos, degrees_avg_pos = link_centrality(degrees, edges)
    #degrees_sub_neg, degrees_avg_neg = link_centrality(degrees, non_edges)

    betweenness = np.array(list(nx.betweenness_centrality(graph_hic, normalized=True).values()))
    #betweenness_sub_pos, betweenness_avg_pos = link_centrality(betweenness, edges)
    #betweenness_sub_neg, betweenness_avg_neg = link_centrality(betweenness, non_edges)

    clustering = np.array(list(nx.clustering(graph_hic).values()))
    #clustering_sub_pos, clustering_avg_pos = link_centrality(clustering, edges)
    #clustering_sub_neg, clustering_avg_neg = link_centrality(clustering, non_edges)

    '''parameters_pos = np.vstack((#shortest_path_lengths_pos, jaccard_index_pos,
                                degrees_sub_pos, degrees_avg_pos,
                                betweenness_sub_pos, betweenness_avg_pos,
                                clustering_sub_pos, clustering_avg_pos))

    parameters_neg = np.vstack((#shortest_path_lengths_neg, jaccard_index_neg,
                                degrees_sub_neg, degrees_avg_neg,
                                betweenness_sub_neg, betweenness_avg_neg,
                                clustering_sub_neg, clustering_avg_neg))'''

    node_features = np.vstack((degrees, betweenness, clustering)).T
    np.save('embeddings/MCF7/topological/chr_02.npy', node_features)

    '''X = np.hstack((parameters_pos, parameters_neg)).T
    y = np.hstack((np.ones(n_edges), np.zeros(n_non_edges)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    rocs = []
    accuracies = []
    for train_index, val_index in skf.split(X_train, y_train):
        # classifier = LogisticRegression(solver='lbfgs')
        classifier = MLPClassifier(max_iter=500)
        classifier.fit(X_train[train_index], y_train[train_index])
        y_pred = classifier.predict(X_train[val_index])
        roc_top = roc_auc_score(y_train[val_index], y_pred)
        accuracy_top = classifier.score(X_train[val_index], y_train[val_index])
        rocs.append(roc_top)
        accuracies.append(accuracy_top)
        print("TOP Accuracy:", accuracy_top, "- ROC:", roc_top)

    print("TOP Mean Accuracy:", np.mean(accuracies), "- Mean ROC:", np.mean(rocs))'''
