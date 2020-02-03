import argparse
import hashlib
import itertools
import os
import pickle
import warnings
from collections import defaultdict
from multiprocessing import Pool

warnings.simplefilter(action='ignore', category=FutureWarning)

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sps
import wandb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from link_prediction.utils import evaluate_embedding, from_scipy_sparse_matrix


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

os.environ["OMP_NUM_THREADS"] = "20"
os.environ["OPENBLAS_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["VECLIB_MAXIMUM_THREADS"] = "20"
os.environ["NUMEXPR_NUM_THREADS"] = "20"


def topological_features(_args, _edges, _non_edges):
    adj_hic = sps.load_npz('data/{}/{}/{}_{}.npz'.format(_args.dataset, _args.folder, _args.folder, _args.name))
    graph_hic = from_scipy_sparse_matrix(adj_hic)
    graph_hic = nx.convert_node_labels_to_integers(graph_hic)

    degrees = np.array(list(dict(graph_hic.degree()).values()))

    betweenness = np.array(list(betweenness_centrality_parallel(graph_hic, processes=25).values()))

    clustering = np.array(list(nx.clustering(graph_hic).values()))

    node_embs = np.vstack((degrees, betweenness, clustering)).T
    np.save('embeddings/embeddings_chr_{:02d}_{:02d}_topological'.format(_args.chr_src, _args.chr_tgt), node_embs)

    parameters_pos = _edges.T
    parameters_neg = _non_edges.T
    # start = time()
    if _args.aggregator == 'concat':
        parameters_pos = np.vstack((degrees[_edges[:, 0]], degrees[_edges[:, 1]],
                                    betweenness[_edges[:, 0]], betweenness[_edges[:, 1]],
                                    clustering[_edges[:, 0]], clustering[_edges[:, 1]]))

        parameters_neg = np.vstack((degrees[_non_edges[:, 0]], degrees[_non_edges[:, 1]],
                                    betweenness[_non_edges[:, 0]], betweenness[_non_edges[:, 1]],
                                    clustering[_non_edges[:, 0]], clustering[_non_edges[:, 1]]))
    else:
        degrees_sub_pos, degrees_avg_pos = link_centrality(degrees, _edges)
        degrees_sub_neg, degrees_avg_neg = link_centrality(degrees, _non_edges)
        betweenness_sub_pos, betweenness_avg_pos = link_centrality(betweenness, _edges)
        betweenness_sub_neg, betweenness_avg_neg = link_centrality(betweenness, _non_edges)
        clustering_sub_pos, clustering_avg_pos = link_centrality(clustering, _edges)
        clustering_sub_neg, clustering_avg_neg = link_centrality(clustering, _non_edges)
        parameters_pos = np.vstack((parameters_pos, degrees_sub_pos,
                                    degrees_avg_pos,
                                    betweenness_sub_pos,
                                    betweenness_avg_pos,
                                    clustering_sub_pos,
                                    clustering_avg_pos))

        parameters_neg = np.vstack((parameters_neg, degrees_sub_neg,
                                    degrees_avg_neg,
                                    betweenness_sub_neg,
                                    betweenness_avg_neg,
                                    clustering_sub_neg,
                                    clustering_avg_neg))

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
    if _args.id_features:
        parameters_pos = np.vstack((parameters_pos, _edges.T))
        parameters_neg = np.vstack((parameters_neg, _non_edges.T))

    # end = time()
    # print("Time for feature generation", end - start)

    X = np.hstack((parameters_pos, parameters_neg)).T
    print(X.shape)
    if _args.edge_features:
        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=9999)
        X = imp.fit_transform(X)
    return X


# ToDo: works only when predicting intra-chromosomal interactions
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PRAD')
    parser.add_argument('--chr-src', type=int, default=2)
    parser.add_argument('--chr-tgt', type=int, default=None)
    parser.add_argument('--n-iter', type=int, default=1)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--method', type=str, default='node2vec',
                        choices=['random', 'distance', 'topological', 'svd', 'node2vec'])

    # ToDo: handle primary, observed, KR
    # parser.add_argument('--interactions', type=str, nargs='*', default=['primary_observed_KR_all_50000_50000_0.9073'])
    parser.add_argument('--resolutions', type=int, nargs='*', default=[40000, 40000])
    parser.add_argument('--windows', type=int, nargs='*', default=[40000, 40000])
    parser.add_argument('--hic-thrs', type=str, nargs='*', default=['3.64', '1.23'])
    parser.add_argument('--weights', type=str, nargs='*', default=[None, None])
    parser.add_argument('--norms', nargs='*', type=str, choices=['NONE', 'VC', 'VC_SQRT', 'KR', 'ICE'],
                        default=['KR', 'KR'])
    parser.add_argument('--types', nargs='*', type=str, choices=['observed', 'oe'], default=['observed', 'observed'])

    parser.add_argument('--coexp-features', default=False, action='store_true')
    parser.add_argument('--edge-features', default=True, action='store_true')
    parser.add_argument('--id-features', default=False, action='store_true')
    parser.add_argument('--aggregator', nargs='*', default=['hadamard'])
    parser.add_argument('--classifier', default='rf', choices=['mlp', 'lr', 'svm', 'mlp_2', 'rf'])
    parser.add_argument('--full-interactions', default=False, action='store_true')
    parser.add_argument('--full-coexpression', default=False, action='store_true')
    parser.add_argument('--zero-median', default=False, action='store_true')
    parser.add_argument('--coexp-thrs', nargs='*', type=str, default=['0.59', '0.75'])
    parser.add_argument('--save-predictions', default=True, action='store_true')
    parser.add_argument('--emb-size', type=int, default=16)
    parser.add_argument('--force', default=False, action='store_true')
    parser.add_argument('--coexp-intra', default=True, action='store_true')
    parser.add_argument('--inter-ratio', type=float, default=0.3)
    parser.add_argument('--test', default=False, action='store_true')

    # Node2vec params
    parser.add_argument('--num-walks', type=int, default=10)
    parser.add_argument('--walk-len', type=int, default=80)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--q', type=float, default=1.0)
    parser.add_argument('--window', type=int, default=10)

    parser.add_argument('--wandb', default=True, action='store_true')
    args = parser.parse_args()

    args.embedding = ''
    if args.method != 'distance':
        args.embedding = 'es{}'.format(args.emb_size)

    if args.method == 'node2vec':
        args.embedding += '_nw{}_wl{}_p{}_q{}'.format(args.num_walks, args.walk_len, args.p, args.q)

    if args.chr_tgt is None:
        args.chr_tgt = args.chr_src

    if args.full_coexpression or args.full_interactions:
        chrs_coexp = 'all'
    else:
        chrs_coexp = '{:02d}_{:02d}'.format(args.chr_src, args.chr_tgt)

    if args.full_interactions:
        chrs_interactions = 'all'
    else:
        chrs_interactions = '{}_{}'.format(args.chr_src, args.chr_tgt)

    # ToDo: add constraint that resolutions, windows and thresholds have to have the same length
    hic_files = ['primary_{}_{}'.format(type, norm) for
                 type, norm in zip(args.types, args.norms)]

    hic_preprocessings = ['{}_{}_{}{}'.format(resolution, window, threshold, ('_' + str(weight)) if weight else '') for
                          resolution, window, threshold, weight in
                          zip(args.resolutions, args.windows, args.hic_thrs, args.weights)]

    print(hic_files)
    print(hic_preprocessings)

    args.interactions = ['{}_{}_{}'.format(file, chrs_interactions, preprocessing) for
                         file, preprocessing in zip(hic_files, hic_preprocessings)]

    interactions_no_chr = ['{}_{}'.format(file, preprocessing) for
                           file, preprocessing in zip(hic_files, hic_preprocessings)]
    args.interactions = '_'.join(args.interactions)

    args.aggregator = '_'.join(args.aggregator)
    coexp_thrs = '_'.join(args.coexp_thrs)

    if args.coexp_features:
        args.folder = 'coexpression'
        args.name = 'chr_{}_{}'.format(chrs_coexp, coexp_thrs)
        experiment_id = '{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, args.classifier, args.n_splits, coexp_thrs,
                                                      args.method,
                                                      args.embedding, args.aggregator)
    else:
        args.folder = 'interactions'
        args.name = args.interactions
        experiment_id = '{}_{}_{}_{}_{}_{}_'.format(args.dataset, args.classifier, args.n_splits, coexp_thrs,
                                                    args.method, '_'.join(interactions_no_chr), args.embedding)
        experiment_id += '_'.join(['{}_{}_{}'.format(resolution, window, threshold) for resolution, window, threshold in
                                   zip(args.resolutions, args.windows, args.hic_thrs)])
        experiment_id += '_' + str(args.aggregator)
    print(experiment_id)

    args.embedding = args.name + '_' + args.embedding

    print(args.name)

    id_hash = str(int(hashlib.sha1(experiment_id.encode()).hexdigest(), 16) % (10 ** 8))

    print(id_hash)

    if not os.path.exists('results/{}/chr_{:02d}'.format(args.dataset, args.chr_src)):
        os.makedirs('results/{}/chr_{:02d}'.format(args.dataset, args.chr_src))

    if not os.path.exists('predictions/{}/chr_{:02d}'.format(args.dataset, args.chr_src)):
        os.makedirs('predictions/{}/chr_{:02d}'.format(args.dataset, args.chr_src))

    if args.method == 'topological':
        if args.full_coexpression:
            filename = 'chr_all/{}_{}_{}_{}{}.pkl'.format(args.classifier,
                                                          args.method, args.name, args.aggregator,
                                                          '_test' if args.test else '')
        else:
            filename = 'chr_{:02d}/{}_{}_{}_{}{}.pkl'.format(args.chr_src, args.classifier,
                                                             args.method, args.name, args.aggregator,
                                                             '_test' if args.test else '')
    else:
        if args.full_coexpression:
            filename = 'chr_all/{}_{}_{}_{}_{}{}.pkl'.format(args.classifier, args.method, args.embedding,
                                                             args.aggregator, coexp_thrs,
                                                             '_test' if args.test else '')
        else:
            filename = 'chr_{:02d}/{}_{}_{}_{}_{}{}.pkl'.format(args.chr_src, args.classifier, args.method,
                                                                args.embedding,
                                                                args.aggregator, coexp_thrs,
                                                                '_test' if args.test else '')
    if not os.path.exists('results/{}/{}'.format(args.dataset, filename)) or args.force:

        coexpression = sps.load_npz(
            'data/{}/coexpression/coexpression_chr_{}_{}.npz'.format(args.dataset, chrs_coexp, coexp_thrs))

        # ToDo: fix for non-interchromosomal
        if len(args.coexp_thrs) > 1:
            coexpression[coexpression == 0] = -1

        degrees = np.ravel((coexpression == 1).sum(axis=0))
        coexpression = sps.triu(coexpression, k=1).toarray()  # .tocsr()

        chr_sizes = np.load(
            '/home/varrone/Prj/gene-expression-chromatin/src_old/coexp_hic_corr/data/{}/chr_sizes.npy'.format(args.dataset))


        disconnected_nodes = np.load(
            '/home/varrone/Prj/gene-expression-chromatin/src_old/coexp_hic_corr/data/{}/disconnected_nodes/{}.npy'.format(
                args.dataset, args.name))

        if args.full_interactions and not args.full_coexpression:
            start_src = int(np.sum(chr_sizes[:args.chr_src]))
            end_src = int(start_src + chr_sizes[args.chr_src])

            start_tgt = int(np.sum(chr_sizes[:args.chr_tgt]))
            end_tgt = int(start_tgt + chr_sizes[args.chr_tgt])

            coexpression = coexpression[start_src:end_src, start_tgt:end_tgt]

            disconnected_nodes_src = disconnected_nodes[
                                         (disconnected_nodes >= start_src) & (disconnected_nodes < end_src)] - start_src
            disconnected_nodes_tgt = disconnected_nodes[
                                         (disconnected_nodes >= start_tgt) & (disconnected_nodes < end_tgt)] - start_tgt
        else:
            disconnected_nodes_src = disconnected_nodes
            disconnected_nodes_tgt = disconnected_nodes

        print("N. disconnected nodes:", len(disconnected_nodes_src))
        if len(disconnected_nodes) > 0:
            coexpression[disconnected_nodes_src] = 0
            coexpression[:, disconnected_nodes_tgt] = 0

        n_nodes = coexpression.shape[0]

        if args.full_coexpression:
            shapes = [np.load(
                '../coexp_hic_corr/data/{}/coexp/coexpression_chr_{:02d}_{:02d}.npy'.format(args.dataset, i, i)).shape for i
                      in
                      range(1, 23)]

            mask = np.zeros(np.sum(shapes, axis=0))

            r, c = 0, 0
            for i, (rr, cc) in enumerate(shapes):
                mask[r:r + rr, c:c + cc] = np.ones((rr, cc))
                r += rr
                c += cc

            coexpression_intra = coexpression * mask
        else:
            coexpression_intra = coexpression
            mask = None

        edges_intra = np.array(np.argwhere(coexpression_intra == 1))
        edges_intra_nodes = np.unique(edges_intra)

        non_nodes_intra = np.setdiff1d(np.arange(n_nodes), edges_intra_nodes)

        coexpression_intra_neg = coexpression_intra.copy()
        coexpression_intra_neg[non_nodes_intra, :] = 0
        coexpression_intra_neg[:, non_nodes_intra] = 0

        non_edges_intra = np.array(np.argwhere(coexpression_intra_neg == -1))
        non_edges_intra = non_edges_intra[
            np.random.choice(non_edges_intra.shape[0], edges_intra.shape[0], replace=False)]

        edges = edges_intra
        non_edges = non_edges_intra

        if not args.coexp_intra:
            coexpression_inter = coexpression * np.logical_not(mask)

            edges_inter = np.array(np.argwhere(coexpression_inter == 1))

            if edges_intra.shape[0] > edges_inter.shape[0]:
                n_edges_inter = edges_inter.shape[0]
            else:
                n_edges_inter = int(edges_intra.shape[0] * args.inter_ratio)
            print('N. intra edges', edges_intra.shape[0], '- N. inter edges ', edges_inter.shape[0], '->',
                  n_edges_inter)
            edges_inter = edges_inter[
                np.random.choice(edges_inter.shape[0], n_edges_inter, replace=False)]
            edges_inter_nodes = np.unique(edges_inter)

            non_nodes_inter = np.setdiff1d(np.arange(n_nodes), edges_inter_nodes)

            coexpression_inter_neg = coexpression_inter.copy()
            coexpression_inter_neg[non_nodes_inter, :] = 0
            coexpression_inter_neg[:, non_nodes_inter] = 0

            non_edges_inter = np.array(np.argwhere(coexpression_inter_neg == -1))
            non_edges_inter = non_edges_inter[
                np.random.choice(non_edges_inter.shape[0], edges_inter.shape[0], replace=False)]

            edges = np.vstack((edges, edges_inter))
            non_edges = np.vstack((non_edges, non_edges_inter))

        n_edges = edges.shape[0]
        n_non_edges = non_edges.shape[0]

        if args.wandb:
            wandb.init(project="coexp-inference-models")
            wandb.config.update({'id': id_hash,
                                 'dataset': args.dataset,
                                 'fold': args.n_splits,
                                 'windows': '_'.join(map(str, args.windows)),
                                 'chr src': args.chr_src,
                                 'chr tgt': args.chr_tgt,
                                 'resolutions': '_'.join(map(str, args.resolutions)),
                                 'hic thresholds': '_'.join(map(str, args.hic_thrs)),
                                 'coexp threshold': coexp_thrs,
                                 'full interactions': args.full_interactions,
                                 'full coexpression': args.full_coexpression,
                                 'embedding method': args.method,
                                 'aggregator': args.aggregator,
                                 'classifier': args.classifier,
                                 'interactions': args.interactions,
                                 'embeddings size': args.emb_size,
                                 'test':args.test})

        if args.method == 'topological':
            X = topological_features(args, edges, non_edges)
        elif args.method == 'ids':
            X = np.vstack((edges, non_edges))
        elif args.method == 'distance':
            if args.full_interactions:
                gene_info = pd.read_csv(
                    '/home/varrone/Prj/gene-expression-chromatin/src_old/coexp_hic_corr/data/{}/rna/{}_chr_all_rna.csv'.format(
                        args.dataset, args.dataset))
            else:
                gene_info = pd.read_csv(
                    '/home/varrone/Prj/gene-expression-chromatin/src_old/coexp_hic_corr/data/{}/rna/{}_chr_{:02d}_rna.csv'.format(
                        args.dataset, args.dataset, args.chr_src))

            pos_distances = np.abs(gene_info.iloc[edges[:, 0]]['Transcription start site (TSS)'].to_numpy() -
                                   gene_info.iloc[edges[:, 1]]['Transcription start site (TSS)'].to_numpy())

            neg_distances = np.abs(gene_info.iloc[non_edges[:, 0]]['Transcription start site (TSS)'].to_numpy() -
                                   gene_info.iloc[non_edges[:, 1]]['Transcription start site (TSS)'].to_numpy())

            pos_features = np.hstack((edges, pos_distances[:, None]))
            neg_features = np.hstack((non_edges, neg_distances[:, None]))
            X = np.vstack((pos_features, neg_features))
        else:
            if args.method == 'random':
                embeddings = np.random.rand(n_nodes, args.emb_size)
                # embeddings = np.ones((n_nodes, 8))
            else:
                print(args.embedding)
                embeddings = np.load(
                    './embeddings/{}/{}/{}.npy'.format(args.dataset, args.method, args.embedding))

            adj = sps.load_npz(
                'data/{}/{}/{}_{}.npz'.format(args.dataset, args.folder, args.folder, args.name))
            adj = adj.todense()
            np.fill_diagonal(adj, 1)

            emb_neigh = np.empty((adj.shape[0], embeddings.shape[1]))

            for i in range(adj.shape[0]):
                if i in disconnected_nodes:
                    continue
                neighbors = np.where(adj[i] == 1)[0]
                emb_neigh[i, :] = np.sum(embeddings[neighbors], axis=0) / neighbors.shape[0]

            if args.full_interactions and not args.full_coexpression:
                embeddings = embeddings[start_src:end_src]

            #ToDo: remove?
            pos_features = edges
            neg_features = non_edges
            if 'hadamard' in args.aggregator:
                pos_features_had = embeddings[edges[:, 0]] * embeddings[edges[:, 1]]
                neg_features_had = embeddings[non_edges[:, 0]] * embeddings[non_edges[:, 1]]
                if pos_features is None or neg_features is None:
                    pos_features = pos_features_had
                    neg_features = neg_features_had
                else:
                    pos_features = np.hstack((pos_features, pos_features_had))
                    neg_features = np.hstack((neg_features, neg_features_had))
            if 'avg' in args.aggregator and 'nwavg' not in args.aggregator:
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
            if 'sub' in args.aggregator and 'nwsub' not in args.aggregator:
                pos_features_sub = np.abs(embeddings[edges[:, 0]] - embeddings[edges[:, 1]])
                neg_features_sub = np.abs(embeddings[non_edges[:, 0]] - embeddings[non_edges[:, 1]])
                if pos_features is None or neg_features is None:
                    pos_features = pos_features_sub
                    neg_features = neg_features_sub
                else:
                    pos_features = np.hstack((pos_features, pos_features_sub))
                    neg_features = np.hstack((neg_features, neg_features_sub))
            if 'l2' in args.aggregator and 'nwl2' not in args.aggregator:
                pos_features_l2 = np.power(embeddings[edges[:, 0]] - embeddings[edges[:, 1]], 2)
                neg_features_l2 = np.power(embeddings[non_edges[:, 0]] - embeddings[non_edges[:, 1]], 2)
                if pos_features is None or neg_features is None:
                    pos_features = pos_features_l2
                    neg_features = neg_features_l2
                else:
                    pos_features = np.hstack((pos_features, pos_features_l2))
                    neg_features = np.hstack((neg_features, neg_features_l2))
            if 'nwhad' in args.aggregator:
                pos_features_nwhad = emb_neigh[edges[:, 0]] * emb_neigh[edges[:, 1]]
                neg_features_nwhad = emb_neigh[non_edges[:, 0]] * emb_neigh[non_edges[:, 1]]
                if pos_features is None or neg_features is None:
                    pos_features = pos_features_nwhad
                    neg_features = neg_features_nwhad
                else:
                    pos_features = np.hstack((pos_features, pos_features_nwhad))
                    neg_features = np.hstack((neg_features, neg_features_nwhad))
            if 'nwavg' in args.aggregator:
                pos_features_nwavg = np.array(
                    list(map(lambda edge: np.mean((emb_neigh[edge[0]], emb_neigh[edge[1]]), axis=0), edges)))
                neg_features_nwavg = np.array(
                    list(map(lambda edge: np.mean((emb_neigh[edge[0]], emb_neigh[edge[1]]), axis=0), non_edges)))

                if pos_features is None or neg_features is None:
                    pos_features = pos_features_nwavg
                    neg_features = neg_features_nwavg
                else:
                    pos_features = np.hstack((pos_features, pos_features_nwavg))
                    neg_features = np.hstack((neg_features, neg_features_nwavg))

            if 'nwl1' in args.aggregator:
                pos_features_nwl1 = np.abs(emb_neigh[edges[:, 0]] - emb_neigh[edges[:, 1]])
                neg_features_nwl1 = np.abs(emb_neigh[non_edges[:, 0]] - emb_neigh[non_edges[:, 1]])
                if pos_features is None or neg_features is None:
                    pos_features = pos_features_nwl1
                    neg_features = neg_features_nwl1
                else:
                    pos_features = np.hstack((pos_features, pos_features_nwl1))
                    neg_features = np.hstack((neg_features, neg_features_nwl1))
            if 'nwl2' in args.aggregator:
                pos_features_nwl2 = np.power(emb_neigh[edges[:, 0]] - emb_neigh[edges[:, 1]], 2)
                neg_features_nwl2 = np.power(emb_neigh[non_edges[:, 0]] - emb_neigh[non_edges[:, 1]], 2)
                if pos_features is None or neg_features is None:
                    pos_features = pos_features_nwl2
                    neg_features = neg_features_nwl2
                else:
                    pos_features = np.hstack((pos_features, pos_features_nwl2))
                    neg_features = np.hstack((neg_features, neg_features_nwl2))
            if 'concat' in args.aggregator and 'nwcat' not in args.aggregator:
                pos_features_cat = np.hstack((embeddings[edges[:, 0]], embeddings[edges[:, 1]]))
                neg_features_cat = np.hstack((embeddings[non_edges[:, 0]], embeddings[non_edges[:, 1]]))
                if pos_features is None or neg_features is None:
                    pos_features = pos_features_cat
                    neg_features = neg_features_cat
                else:
                    pos_features = np.hstack((pos_features, pos_features_cat))
                    neg_features = np.hstack((neg_features, neg_features_cat))
            if 'nwcat' in args.aggregator:
                pos_features_cat = np.hstack((emb_neigh[edges[:, 0]], emb_neigh[edges[:, 1]]))
                neg_features_cat = np.hstack((emb_neigh[non_edges[:, 0]], emb_neigh[non_edges[:, 1]]))
                if pos_features is None or neg_features is None:
                    pos_features = pos_features_cat
                    neg_features = neg_features_cat
                else:
                    pos_features = np.hstack((pos_features, pos_features_cat))
                    neg_features = np.hstack((neg_features, neg_features_cat))
            X = np.vstack((pos_features, neg_features))
        y = np.hstack((np.ones(n_edges), np.zeros(n_non_edges)))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

        results = defaultdict(list)
        if args.test:
            results = evaluate_embedding(X_train, y_train, args.classifier, verbose=1, clf_params={'n_estimators': 100},
                                         mask=mask, X_test=X_test, y_test=y_test)
            wandb.run.summary.update({'accuracy': results['acc'],
                                      'accuracy std': results['acc'],
                                      'precision': results['precision'],
                                      'precision std': results['precision'],
                                      'recall': results['recall'],
                                      'recall std': results['recall']})
        else:
            for i in range(args.n_iter):
                results_iter = evaluate_embedding(X_train, y_train, args.classifier, n_iter=args.n_iter, verbose=1,
                                                  clf_params={'n_estimators': 100}, n_splits=args.n_splits, mask=mask)
                for key in results_iter.keys():
                    results[key].extend(results_iter[key])

            for key in results.keys():
                wandb.run.summary.update({'accuracy': np.mean(results['acc']),
                                          'accuracy std': np.std(results['acc']),
                                          'precision': np.mean(results['precision']),
                                          'precision std': np.std(results['precision']),
                                          'recall': np.mean(results['recall']),
                                          'recall std': np.std(results['recall'])})

        with open('results/{}/{}'.format(args.dataset, filename), 'wb') as file_save:
            pickle.dump(results, file_save)

        wandb.save('results/{}/{}'.format(args.dataset, filename))

        print("Mean Accuracy:", np.mean(results['acc']), "- Mean ROC:", np.mean(results['roc']), "- Mean F1:",
              np.mean(results['f1']),
              "- Mean Precision:", np.mean(results['precision']), "- Mean Recall", np.mean(results['recall']))
    else:
        print('Result already computed for {}. Skipped.'.format(filename))
