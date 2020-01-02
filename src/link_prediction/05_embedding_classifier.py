import argparse
import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from link_prediction.utils import evaluate_embedding


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
    adj_hic = np.load(
        'data/{}/{}/{}_{}.npy'.format(_args.dataset, _args.folder, _args.folder, _args.name))
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
    np.save('embeddings/embeddings_chr_{:02d}_{:02d}_topological'.format(_args.chr_src, _args.chr_tgt), node_embs)

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
    if args.id_features:
        parameters_pos = np.vstack((parameters_pos, _edges.T))
        parameters_neg = np.vstack((parameters_neg, _non_edges.T))

    X = np.hstack((parameters_pos, parameters_neg)).T
    print(X.shape)
    if _args.edge_features:
        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=9999)
        X = imp.fit_transform(X)
    return X


# ToDo: works only when predicting intra-chromosomal interactions
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MCF7')
    parser.add_argument('--chr-src', type=int, default=1)
    parser.add_argument('--chr-tgt', type=int, default=None)
    parser.add_argument('--n-iter', type=int, default=2)
    parser.add_argument('--embedding', type=str, default='es8_nw10_wl80_p1.0_q1.0')
    parser.add_argument('--method', type=str, default='node2vec')
    parser.add_argument('--interactions', type=str, default='primary_observed_KR_all_50000_50000_0.9073')
    parser.add_argument('--coexpression', type=str, default=None)
    parser.add_argument('--edge-features', default=True, action='store_true')
    parser.add_argument('--id-features', default=False, action='store_true')
    parser.add_argument('--aggregator', default='concat', nargs='*')
    parser.add_argument('--classifier', default='rf', choices=['mlp', 'lr', 'svm', 'mlp_2', 'rf'])
    parser.add_argument('--full-interactions', default=True, action='store_true')
    parser.add_argument('--full-coexpression', default=True, action='store_true')
    parser.add_argument('--zero-median', default=False, action='store_true')
    parser.add_argument('--threshold', type=float, default=0.4113)
    args = parser.parse_args()

    if args.coexpression:
        args.folder = 'coexpression'
        args.name = args.coexpression
    else:
        args.folder = 'interactions'
        args.name = args.interactions

    if args.chr_tgt is None:
        args.chr_tgt = args.chr_src

    if args.full_coexpression:
        chrs = 'all'
    else:
        chrs = str(args.chr_src) + '_' + str(args.chr_tgt)

    if type(args.aggregator) == list:
        args.aggregator = '_'.join(args.aggregator)

    args.embedding = args.interactions + '_' + args.embedding

    coexpression = sps.load_npz(
        'data/{}/coexpression/coexpression_chr_{}_{}.npz'.format(args.dataset, chrs, args.threshold))

    print(args.name)

    chr_sizes = np.load(
        '/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/chr_sizes.npy'.format(args.dataset))

    disconnected_nodes = np.load(
        '/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/disconnected_nodes/{}.npy'.format(
            args.dataset, args.name))

    if args.full_interactions and not args.full_coexpression:
        start_src = np.cumsum(chr_sizes[:args.chr_src])
        end_src = start_src + chr_sizes[args.chr_src]

        start_tgt = np.cumsum(chr_sizes[:args.chr_tgt])
        end_tgt = start_tgt + chr_sizes[args.chr_tgt]

        coexpression = coexpression[start_src:end_src, start_tgt:end_tgt]

        disconnected_nodes_src = disconnected_nodes[start_src <= disconnected_nodes < end_src] - start_src
        disconnected_nodes_tgt = disconnected_nodes[start_tgt <= disconnected_nodes < end_tgt] - start_tgt
    else:
        disconnected_nodes_src = disconnected_nodes
        disconnected_nodes_tgt = disconnected_nodes

    coexpression[disconnected_nodes_src] = 0
    coexpression[:, disconnected_nodes_tgt] = 0

    edges = np.array(np.argwhere(coexpression == 1))
    n_edges = edges.shape[0]

    n_nodes = coexpression.shape[0]
    edges_nodes = np.unique(edges)
    non_nodes = np.setdiff1d(np.arange(n_nodes), edges_nodes)

    print("N. non nodes:", non_nodes.shape[0])

    coexpression_neg = coexpression.copy()

    coexpression_neg[non_nodes, :] = 0
    coexpression_neg[:, non_nodes] = 0
    non_edges = np.array(np.argwhere(coexpression_neg == -1))

    non_edges = non_edges[np.random.choice(non_edges.shape[0], n_edges, replace=False)]
    n_non_edges = non_edges.shape[0]

    n_nodes = coexpression.shape[0]
    if args.method == 'topological':
        X = topological_features(args, edges, non_edges)
    elif args.method == 'ids':
        X = np.vstack((edges, non_edges))
    elif args.method == 'distance':
        gene_info = pd.read_csv(
            '/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/rna/{}_chr_{:02d}_rna.csv'.format(
                args.dataset, args.dataset, args.chr_src))

        pos_distances = np.abs(gene_info.iloc[edges[:, 0]]['Transcription start site (TSS)'].to_numpy() -
                               gene_info.iloc[edges[:, 1]]['Transcription start site (TSS)'].to_numpy())

        neg_distances = np.abs(gene_info.iloc[non_edges[:, 0]]['Transcription start site (TSS)'].to_numpy() -
                               gene_info.iloc[non_edges[:, 1]]['Transcription start site (TSS)'].to_numpy())

        pos_features = pos_distances[:, None]
        neg_features = neg_distances[:, None]
        X = np.vstack((pos_features, neg_features))
    else:
        if args.method == 'random':
            embeddings = np.random.rand(n_nodes, 8)
        else:
            embeddings = np.load(
                './embeddings/{}/{}/{}.npy'.format(args.dataset, args.method, args.embedding))

        pos_features = None
        neg_features = None
        if 'hadamard' in args.aggregator:
            pos_features = np.array(list(map(lambda edge: embeddings[edge[0]] * embeddings[edge[1]], edges)))
            neg_features = np.array(list(map(lambda edge: embeddings[edge[0]] * embeddings[edge[1]], non_edges)))
        if 'avg' in args.aggregator:
            pos_features_avg = np.array(
                list(map(lambda edge: np.mean((embeddings[edge[0]], embeddings[edge[1]]), axis=0), edges)))
            neg_features_avg = np.array(
                list(map(lambda edge: np.mean((embeddings[edge[0]], embeddings[edge[1]]), axis=0), non_edges)))
            if pos_features is None or neg_features is None:
                pos_features = pos_features_avg
                neg_features = neg_features_avg
            else:
                pos_features = np.hstack((pos_features, pos_features_avg))
                neg_features = np.hstack((neg_features, neg_features_avg))
        if 'concat' in args.aggregator:
            pos_features_cat = np.hstack((embeddings[edges[:,0]], embeddings[edges[:, 1]]))
            neg_features_cat = np.hstack((embeddings[non_edges[:,0]], embeddings[non_edges[:, 1]]))
            if pos_features is None or neg_features is None:
                pos_features = pos_features_cat
                neg_features = neg_features_cat
            else:
                pos_features = np.hstack((pos_features, pos_features_cat))
                neg_features = np.hstack((neg_features, neg_features_cat))
        X = np.vstack((pos_features, neg_features))
    y = np.hstack((np.ones(n_edges), np.zeros(n_non_edges)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    print(X_train.shape)
    results = evaluate_embedding(X_train, y_train, args.classifier, n_iter=args.n_iter, verbose=1,
                                 clf_params={'n_estimators': 100})

    if not os.path.exists('results/{}/chr_{:02d}'.format(args.dataset, args.chr_src)):
        os.makedirs('results/{}/chr_{:02d}'.format(args.dataset, args.chr_src))
    if args.method == 'topological':
        with open(
                'results/{}/chr_{:02d}/{}_{}_{}_{}{}.pkl'.format(args.dataset, args.chr_src, args.classifier,
                                                                 args.method, args.name, args.aggregator,
                                                                 '_zero_median' if args.zero_median else ''),
                'wb') as file_save:
            pickle.dump(results, file_save)
    else:
        with open('results/{}/chr_{:02d}/{}_{}_{}{}.pkl'.format(args.dataset, args.chr_src,
                                                                                 args.classifier, args.method,
                                                                                 args.embedding,
                                                                                 args.aggregator,
                                                                                 '_zero_median' if args.zero_median else ''),
                  'wb') as file_save:
            pickle.dump(results, file_save)

    print("Mean Accuracy:", np.mean(results['acc']), "- Mean ROC:", np.mean(results['roc']), "- Mean F1:",
          np.mean(results['f1']),
          "- Mean Precision:", np.mean(results['precision']), "- Mean Recall", np.mean(results['recall']))
