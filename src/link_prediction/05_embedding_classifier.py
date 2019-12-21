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
    adj_hic = np.load('data/{}/interactions/interactions_{}.npy'.format(_args.dataset, _args.interactions))
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

    X = np.hstack((parameters_pos, parameters_neg)).T
    print(X.shape)
    if _args.edge_features:
        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=9999)
        X = imp.fit_transform(X)
    return X


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MCF7')
    parser.add_argument('--chr-src', type=int, default=1)
    parser.add_argument('--chr-tgt', type=int, default=1)
    parser.add_argument('--n-iter', type=int, default=2)
    parser.add_argument('--embedding-pt1', type=str, default='primary_observed_KR')
    parser.add_argument('--embedding-pt2', type=str, default='50000_50000_0.9073_es8')
    parser.add_argument('--method', type=str, default='distance')
    parser.add_argument('--interactions', type=str,
                        default=None)
    parser.add_argument('--edge-features', default=True, action='store_true')
    parser.add_argument('--distance-feature', default=False, action='store_true')
    parser.add_argument('--aggregator', default='concat', choices=['hadamard', 'concat'])
    parser.add_argument('--classifier', default='mlp', choices=['mlp', 'lr', 'svm', 'mlp_2'])
    parser.add_argument('--threshold', type=float, default=0.28)
    parser.add_argument('--inter', default=False, action='store_true')
    parser.add_argument('--genes-chr', default=False, action='store_true')
    parser.add_argument('--zero-median', default=False, action='store_true')
    args = parser.parse_args()

    coexpression = np.load(
        'data/{}/coexpression/coexpression_chr_{:02d}_{:02d}_{}{}.npy'.format(args.dataset, args.chr_src, args.chr_tgt,
                                                                              args.threshold,
                                                                              '_zero_median' if args.zero_median else '', ))

    if args.interactions:
        # ToDo: fix bug if the first gene of the chromosome is disconnected
        genes_chr = np.load(
            '/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/genes_chr/{}.npy'.format(
                args.dataset, args.interactions))
        genes_src = np.where(genes_chr == args.chr_src)[0]
        genes_src_offset = genes_src - np.min(genes_src)
        genes_tgt = np.where(genes_chr == args.chr_tgt)[0]
        genes_tgt_offset = genes_tgt - np.min(genes_tgt)
        coexpression = coexpression[genes_src_offset[:, None], genes_tgt_offset]

    if coexpression.shape[0] != coexpression.shape[1]:
        graph_coexp = nx.algorithms.bipartite.from_biadjacency_matrix(sps.csr_matrix(coexpression))
    else:
        graph_coexp = nx.from_numpy_array(coexpression)

    edges = np.array(list(graph_coexp.edges))
    n_edges = edges.shape[0]
    print('N. edges', n_edges)
    plt.imshow(coexpression, cmap='Oranges')
    plt.show()

    if coexpression.shape[0] != coexpression.shape[1]:
        src_nodes = np.arange(coexpression.shape[0])
        tgt_nodes = np.arange(coexpression.shape[0], coexpression.shape[0] + coexpression.shape[1])
        non_edges = list()
        while len(non_edges) < n_edges:
            src = np.random.choice(src_nodes)
            tgt = np.random.choice(tgt_nodes)
            if not graph_coexp.has_edge(src, tgt):
                non_edges.append((src, tgt))
        non_edges = np.array(non_edges)
        n_non_edges = len(non_edges)

        n_nodes = coexpression.shape[0] + coexpression.shape[1]

    else:
        non_edges = np.array(list(nx.non_edges(graph_coexp)))
        non_edges = non_edges[np.random.choice(non_edges.shape[0], n_edges, replace=False)]
        n_non_edges = non_edges.shape[0]

        n_nodes = coexpression.shape[0]

    adj = np.zeros((n_nodes, n_nodes))
    adj[edges[:, 0], edges[:, 1]] = 1
    adj[non_edges[:, 0], non_edges[:, 1]] = -1

    plt.imshow(adj, cmap='seismic')
    plt.show()

    if args.method == 'topological':
        X = topological_features(args, edges, non_edges)
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
        if args.genes_chr:
            embeddings = np.load(
                './embeddings/{}/{}/{}_{}_{}.npy'.format(args.dataset, args.method, args.embedding_pt1, 'all',
                                                         args.embedding_pt2))
            # ToDo: check inter-chromosomal prediction case
            embeddings = embeddings[genes_src]
            adj = np.dot(embeddings, embeddings.T)

            plt.imshow(adj, cmap='Oranges')
            plt.show()
        else:
            embeddings = np.load(
                './embeddings/{}/{}/{}_{}_{}_{}.npy'.format(args.dataset, args.method, args.embedding_pt1, args.chr_src,
                                                            args.chr_tgt, args.embedding_pt2))

        if args.aggregator == 'hadamard':
            pos_features = np.array(list(map(lambda edge: embeddings[edge[0]] * embeddings[edge[1]], edges)))
            neg_features = np.array(list(map(lambda edge: embeddings[edge[0]] * embeddings[edge[1]], non_edges)))

            if args.distance_feature:
                gene_info = pd.read_csv(
                    '/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/rna/{}_chr_{:02d}_{:02d}_rna.csv'.format(
                        args.dataset, args.dataset, args.chr_src, args.chr_tgt))

                pos_distances = np.abs(gene_info.iloc[edges[:, 0]]['Transcription start site (TSS)'].to_numpy() -
                                       gene_info.iloc[edges[:, 1]]['Transcription start site (TSS)'].to_numpy())

                neg_distances = np.abs(gene_info.iloc[non_edges[:, 0]]['Transcription start site (TSS)'].to_numpy() -
                                       gene_info.iloc[non_edges[:, 1]]['Transcription start site (TSS)'].to_numpy())

                pos_features = np.hstack((pos_features, pos_distances[:, None]))
                neg_features = np.hstack((neg_features, neg_distances[:, None]))

        else:
            pos_features = np.array(
                list(map(lambda edge: np.hstack((embeddings[edge[0]], embeddings[edge[1]])), edges)))
            neg_features = np.array(
                list(map(lambda edge: np.hstack((embeddings[edge[0]], embeddings[edge[1]])), non_edges)))
        X = np.vstack((pos_features, neg_features))
    y = np.hstack((np.ones(n_edges), np.zeros(n_non_edges)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    print(X_train.shape)
    results = evaluate_embedding(X_train, y_train, args.classifier, n_iter=args.n_iter, verbose=1)

    if not os.path.exists('results/{}/chr_{:02d}'.format(args.dataset, args.chr_src)):
        os.makedirs('results/{}/chr_{:02d}'.format(args.dataset, args.chr_src))
    if args.method == 'topological':
        with open(
                'results/{}/chr_{:02d}/{}_{}_{}{}.pkl'.format(args.dataset, args.chr_src, args.classifier, args.method,
                                                              args.interactions,
                                                              '_zero_median' if args.zero_median else '', ),
                'wb') as file_save:
            pickle.dump(results, file_save)
    else:
        with open('results/{}/chr_{:02d}/{}_{}_{}_{:02d}_{:02d}_{}{}.pkl'.format(args.dataset, args.chr_src,
                                                                                 args.classifier, args.method,
                                                                                 args.embedding_pt1, args.chr_src,
                                                                                 args.chr_tgt, args.embedding_pt2,
                                                                                 '_zero_median' if args.zero_median else '', ),
                  'wb') as file_save:
            pickle.dump(results, file_save)

    print("Mean Accuracy:", np.mean(results['acc']), "- Mean ROC:", np.mean(results['roc']), "- Mean F1:",
          np.mean(results['f1']),
          "- Mean Precision:", np.mean(results['precision']), "- Mean Recall", np.mean(results['recall']))
