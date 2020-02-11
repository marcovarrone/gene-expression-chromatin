import itertools
import os
from collections import defaultdict
from multiprocessing import Pool
from time import time

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sps
from bionev.utils import load_embedding
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def select_classifier(classifier_name, clf_params, seed=42):
    if classifier_name == 'mlp':
        classifier = MLPClassifier(max_iter=500, hidden_layer_sizes=(100,))
    elif classifier_name == 'svm':
        classifier = SVC(gamma='scale')
    elif classifier_name == 'rf':
        classifier = RandomForestClassifier(n_jobs=10, **clf_params)
    else:
        classifier = LogisticRegression(solver='lbfgs')
    return classifier


def confusion_matrix_distinct(y_true, y_pred, ids, mask):
    is_intra = mask[ids[:, 0], ids[:, 1]].astype(bool)
    y_true_intra, y_pred_intra = y_true[is_intra], y_pred[is_intra]
    print('Intra accuracy: ', accuracy_score(y_true_intra, y_pred_intra))
    print(confusion_matrix(y_true_intra, y_pred_intra))

    if (is_intra == 0).any():
        y_true_inter, y_pred_inter = y_true[~is_intra], y_pred[~is_intra]
        print('Inter accuracy: ', accuracy_score(y_true_inter, y_pred_inter))
        print(confusion_matrix(y_true_inter, y_pred_inter))


def evaluate(X_train, y_train, X_test, y_test, classifier, mask):
    ids = X_test[:, :2].astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[:, 2:], y_train)
    X_test_scaled = scaler.transform(X_test[:, 2:])

    start = time()
    classifier.fit(X_train_scaled, y_train)
    y_pred = classifier.predict(X_test_scaled)
    end = time()

    if mask is not None:
        confusion_matrix_distinct(y_test, y_pred, ids, mask)

    results = {}
    results['roc'] = roc_auc_score(y_test, y_pred)
    results['acc'] = classifier.score(X_test_scaled, y_test)
    results['f1'] = f1_score(y_test, y_pred)
    results['precision'] = precision_score(y_test, y_pred)
    results['recall'] = recall_score(y_test, y_pred)
    results['predictions'] = list(y_pred)
    return results


def evaluate_embedding(X_train, y_train, classifier_name, verbose=1, clf_params={}, cv_splits=5, mask=None, X_test=None,
                       y_test=None):
    results = defaultdict(list)
    if X_test is None and y_test is None:
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True)
        for train_index, val_index in skf.split(X_train, y_train):
            classifier = select_classifier(classifier_name, clf_params)
            results_iter = evaluate(X_train[train_index], y_train[train_index], X_train[val_index],
                                    y_train[val_index], classifier, mask)
            if verbose:
                print("Accuracy:", results_iter['acc'], "- ROC:", results_iter['roc'], "- F1:", results_iter['f1'],
                      "- Precision:", results_iter['precision'], "- Recall", results_iter['recall'])

            for key in results_iter.keys():
                results[key].append(results_iter[key])
    else:
        classifier = select_classifier(classifier_name, clf_params)
        results = evaluate(X_train, y_train, X_test, y_test, classifier, mask)

        if verbose:
            print("Accuracy:", results['acc'], "- ROC:", results['roc'], "- F1:", results['f1'],
                  "- Precision:", results['precision'], "- Recall", results['recall'])

    return results


def generate_embedding(args, emb_path, interactions_path, command):
    os.makedirs('../../data/{}/embeddings/{}/'.format(args.dataset, args.method.lower()), exist_ok=True)
    if not os.path.exists(
            '../../data/{}/embeddings/{}/{}.npy'.format(args.dataset, args.method.lower(), emb_path)) or args.force:
        adj = np.load(interactions_path)
        graph = from_numpy_matrix(adj)

        nx.write_edgelist(graph,
                                   '../../data/{}/chromatin_networks/{}.edgelist'.format(args.dataset, args.name))

        print(command)
        os.system(command)
        emb_dict = load_embedding(
            '../../data/{}/embeddings/{}/{}.txt'.format(
                args.dataset, args.method.lower(), emb_path))

        emb = np.zeros((adj.shape[0], args.emb_size))

        disconnected_nodes = []

        print('N. genes', adj.shape[0])
        for gene in range(adj.shape[0]):
            try:
                emb[gene, :] = emb_dict[str(gene)]
            except KeyError:
                print('Node', gene, 'disconnected.')
                # np.delete(emb, i, axis=0)
                emb[gene, :] = np.nan
                disconnected_nodes.append(gene)

        os.makedirs('../../data/{}/disconnected_nodes/'.format(args.dataset), exist_ok=True)
        np.save(
            '../../data/{}/disconnected_nodes/{}.npy'.format(
                args.dataset, args.name), np.array(disconnected_nodes))

        if args.save_emb:
            np.save('../../data/{}/embeddings/{}/{}.npy'.format(args.dataset, args.method.lower(), emb_path), emb)
        os.remove('../../data/{}/embeddings/{}/{}.txt'.format(args.dataset, args.method.lower(), emb_path))
        os.remove('../../data/{}/chromatin_networks/{}.edgelist'.format(args.dataset, args.name))
        return emb


def from_numpy_matrix(A):
    # IMPORTANT: do not use for the co-expression matrix, otherwise the nans will be ignored and considered as non_edges
    A[np.isnan(A)] = 0

    if A.shape[0] != A.shape[1]:
        graph = nx.algorithms.bipartite.from_biadjacency_matrix(sps.csr_matrix(A))
    else:
        graph = nx.from_numpy_array(A)
    return graph


def link_centrality(centrality, edges, type):
    centrality_src = centrality[edges[:, 0]]
    centrality_tgt = centrality[edges[:, 1]]
    if type == 'l1':
        centrality = np.abs(centrality_src - centrality_tgt)
    elif type == 'avg':
        centrality = np.mean(np.vstack((centrality_src, centrality_tgt)), axis=0)
    else:
        raise ValueError()
    return centrality


def topological_features(args, edges, non_edges):
    adj_hic = np.load('../../data/{}/chromatin_networks/{}.npy'.format(args.dataset, args.name))
    graph_hic = from_numpy_matrix(adj_hic)
    graph_hic = nx.convert_node_labels_to_integers(graph_hic)

    degrees = np.array(list(dict(graph_hic.degree()).values()))

    betweenness = np.array(list(betweenness_centrality_parallel(graph_hic, 20).values()))

    clustering = np.array(list(nx.clustering(graph_hic).values()))

    node_embs = np.vstack((degrees, betweenness, clustering)).T

    os.makedirs('../../data/{}/embeddings/topological/'.format(args.dataset), exist_ok=True)
    np.save('../../data/{}/embeddings/topological/{}.npy'.format(args.dataset, args.name), node_embs)

    parameters_pos = edges.T
    parameters_neg = non_edges.T
    if 'concat' in args.aggregators:
        parameters_pos = np.vstack((parameters_pos, degrees[edges[:, 0]], degrees[edges[:, 1]],
                                    betweenness[edges[:, 0]], betweenness[edges[:, 1]],
                                    clustering[edges[:, 0]], clustering[edges[:, 1]]))

        parameters_neg = np.vstack((parameters_neg, degrees[non_edges[:, 0]], degrees[non_edges[:, 1]],
                                    betweenness[non_edges[:, 0]], betweenness[non_edges[:, 1]],
                                    clustering[non_edges[:, 0]], clustering[non_edges[:, 1]]))
    if 'avg' in args.aggregators:
        degrees_avg_pos = link_centrality(degrees, edges, 'avg')
        degrees_avg_neg = link_centrality(degrees, non_edges, 'avg')

        betweenness_avg_pos = link_centrality(betweenness, edges, 'avg')
        betweenness_avg_neg = link_centrality(betweenness, non_edges, 'avg')

        clustering_avg_pos = link_centrality(clustering, edges, 'avg')
        clustering_avg_neg = link_centrality(clustering, non_edges, 'avg')

        parameters_pos = np.vstack((parameters_pos, degrees_avg_pos, betweenness_avg_pos, clustering_avg_pos))
        parameters_neg = np.vstack((parameters_neg, degrees_avg_neg, betweenness_avg_neg, clustering_avg_neg))

    if 'l1' in args.aggregators:
        degrees_l1_pos = link_centrality(degrees, edges, 'l1')
        degrees_l1_neg = link_centrality(degrees, non_edges, 'l1')

        betweenness_l1_pos = link_centrality(betweenness, edges, 'l1')
        betweenness_l1_neg = link_centrality(betweenness, non_edges, 'l1')

        clustering_l1_pos = link_centrality(clustering, edges, 'l1')
        clustering_l1_neg = link_centrality(clustering, non_edges, 'l1')

        parameters_pos = np.vstack((parameters_pos, degrees_l1_pos, betweenness_l1_pos, clustering_l1_pos))
        parameters_neg = np.vstack((parameters_neg, degrees_l1_neg, betweenness_l1_neg, clustering_l1_neg))

    if args.edge_features:
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

        parameters_pos = np.vstack((parameters_pos, shortest_path_lengths_pos, jaccard_index_pos))
        parameters_neg = np.vstack((parameters_neg, shortest_path_lengths_neg, jaccard_index_neg))

    X = np.hstack((parameters_pos, parameters_neg)).T
    print(X.shape)

    if args.edge_features:
        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=9999)
        X = imp.fit_transform(X)
    return X


def distance_embedding(dataset, edges, non_edges, chr_src=None):
    if chr_src is None:
        gene_info = pd.read_csv(
            '../../data/{}/rna/expression_info_chr_all_rna.csv'.format(
                dataset))
    else:
        gene_info = pd.read_csv(
            '../../data/{}/rna/expression_info_chr_{}_rna.csv'.format(
                dataset, chr_src))

    pos_distances = np.abs(gene_info.iloc[edges[:, 0]]['Transcription start site (TSS)'].to_numpy() -
                           gene_info.iloc[edges[:, 1]]['Transcription start site (TSS)'].to_numpy())

    neg_distances = np.abs(gene_info.iloc[non_edges[:, 0]]['Transcription start site (TSS)'].to_numpy() -
                           gene_info.iloc[non_edges[:, 1]]['Transcription start site (TSS)'].to_numpy())

    pos_features = np.hstack((edges, pos_distances[:, None]))
    neg_features = np.hstack((non_edges, neg_distances[:, None]))
    X = np.vstack((pos_features, neg_features))
    return X


def method_embedding(args, n_nodes, edges, non_edges, start_src=None, end_src=None):
    if args.method == 'random':
        embeddings = np.random.rand(n_nodes, args.emb_size)
    else:
        #ToDo: if embeddings doesn't exist, run the embedding method
        embeddings = np.load(
            '../../data/{}/embeddings/{}/{}.npy'.format(args.dataset, args.method, args.embedding))

    # Add edges and non_edges ids in dataset to identify the type of interaction for the confusion matrix
    # They will be removed from the dataset before training
    pos_features = edges
    neg_features = non_edges
    if 'hadamard' in args.aggregators:
        pos_features, neg_features = hadamard_embedding(pos_features, neg_features, embeddings, edges,
                                                        non_edges)
    if 'avg' in args.aggregators:
        pos_features, neg_features = average_embedding(pos_features, neg_features, embeddings, edges, non_edges)
    if 'l1' in args.aggregators:
        pos_features, neg_features = l1_embedding(pos_features, neg_features, embeddings, edges, non_edges)
    if 'l2' in args.aggregators:
        pos_features, neg_features = l2_embedding(pos_features, neg_features, embeddings, edges, non_edges)
    if 'concat' in args.aggregators:
        pos_features, neg_features = concat_embedding(pos_features, neg_features, embeddings, edges, non_edges)

    if pos_features is None or neg_features is None:
        raise ValueError('No aggregator defined.')
    X = np.vstack((pos_features, neg_features))
    return X


def append_features(pos_features, neg_features, pos_features_partial, neg_features_partial):
    if pos_features is None or neg_features is None:
        pos_features = pos_features_partial
        neg_features = neg_features_partial
    else:
        pos_features = np.hstack((pos_features, pos_features_partial))
        neg_features = np.hstack((neg_features, neg_features_partial))
    return pos_features, neg_features


def hadamard_embedding(pos_features, neg_features, embeddings, edges, non_edges):
    pos_features_partial = embeddings[edges[:, 0]] * embeddings[edges[:, 1]]
    neg_features_partial = embeddings[non_edges[:, 0]] * embeddings[non_edges[:, 1]]
    return append_features(pos_features, neg_features, pos_features_partial, neg_features_partial)


def average_embedding(pos_features, neg_features, embeddings, edges, non_edges):
    pos_features_partial = np.array(
        list(map(lambda edge: np.mean((embeddings[edge[0]], embeddings[edge[1]]), axis=0), edges)))
    neg_features_partial = np.array(
        list(map(lambda edge: np.mean((embeddings[edge[0]], embeddings[edge[1]]), axis=0), non_edges)))
    return append_features(pos_features, neg_features, pos_features_partial, neg_features_partial)


def l1_embedding(pos_features, neg_features, embeddings, edges, non_edges):
    pos_features_partial = np.abs(embeddings[edges[:, 0]] - embeddings[edges[:, 1]])
    neg_features_partial = np.abs(embeddings[non_edges[:, 0]] - embeddings[non_edges[:, 1]])
    return append_features(pos_features, neg_features, pos_features_partial, neg_features_partial)


def l2_embedding(pos_features, neg_features, embeddings, edges, non_edges):
    pos_features_partial = np.power(embeddings[edges[:, 0]] - embeddings[edges[:, 1]], 2)
    neg_features_partial = np.power(embeddings[non_edges[:, 0]] - embeddings[non_edges[:, 1]], 2)
    return append_features(pos_features, neg_features, pos_features_partial, neg_features_partial)


def concat_embedding(pos_features, neg_features, embeddings, edges, non_edges):
    pos_features_partial = np.hstack((embeddings[edges[:, 0]], embeddings[edges[:, 1]]))
    neg_features_partial = np.hstack((embeddings[non_edges[:, 0]], embeddings[non_edges[:, 1]]))
    return append_features(pos_features, neg_features, pos_features_partial, neg_features_partial)


def chunks(l, n):
    """Divide a list of nodes `l` in `n` chunks"""
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x


def betweenness_centrality_parallel(G, processes=None):
    """Parallel betweenness centrality  function"""
    p = Pool(processes=processes)
    node_divisor = len(p._pool) * 4
    node_chunks = list(chunks(G.nodes(), int(G.order() / node_divisor)))
    num_chunks = len(node_chunks)
    bt_sc = p.starmap(
        nx.betweenness_centrality_source,
        zip([G] * num_chunks, [True] * num_chunks, [None] * num_chunks, node_chunks),
    )

    # Reduce the partial solutions
    bt_c = bt_sc[0]
    for bt in bt_sc[1:]:
        for n in bt:
            bt_c[n] += bt[n]
    return bt_c
