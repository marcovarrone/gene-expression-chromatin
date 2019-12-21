import argparse
import os

import networkx as nx
import numpy as np
import scipy.sparse as sps
from bionev.utils import load_embedding

from link_prediction.utils import set_n_threads

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MCF7')
    parser.add_argument('--interactions', type=str, default='primary_oe_NONE_1_1_10000_40000_sum_0.0')
    parser.add_argument('--emb-size', type=int, default=8)
    parser.add_argument('--n-jobs', type=int, default=10)
    parser.add_argument('--num-walks', type=int, default=10)
    parser.add_argument('--walk-len', type=int, default=80)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--q', type=float, default=1.0)

    parser.add_argument('--all-regions', default=False, action='store_true')
    parser.add_argument('--force', default=False, action='store_true')
    parser.add_argument('--method', default='node2vec', type=str, choices=['node2vec', 'struc2vec'])
    parser.add_argument('--weighted', type=str, default='False')
    parser.add_argument('--save-emb', default=False, action='store_true')
    args = parser.parse_args()

    set_n_threads(args.n_jobs)

    interactions_path = './data/{}/interactions/interactions_{}.npy'.format(
        args.dataset, args.interactions)

    emb_path = '{}_es{}_nw{}_wl{}_p{}_q{}_prova'.format(args.interactions, args.emb_size, args.num_walks, args.walk_len, args.p, args.q)
    if not os.path.exists('embeddings/{}/{}/{}.npy'.format(args.dataset, args.method, emb_path)) or args.force:
        adj = np.load(interactions_path)
        if adj.shape[0] != adj.shape[1]:
            # adj = np.block([[np.zeros((adj.shape[0], adj.shape[0])), adj], [adj.T, np.zeros((adj.shape[1], adj.shape[1]))]])
            graph = nx.algorithms.bipartite.from_biadjacency_matrix(sps.csr_matrix(adj))
        else:
            graph = nx.from_numpy_array(adj)

        if args.weighted == 'True':
            nx.write_weighted_edgelist(graph,
                                       'data/{}/interactions/{}.edgelist'.format(args.dataset, args.interactions))
        else:
            nx.write_edgelist(graph, 'data/{}/interactions/{}.edgelist'.format(args.dataset, args.interactions))

        command = 'bionev --input data/{}/interactions/{}.edgelist '.format(args.dataset, args.interactions) + \
                  '--output ./embeddings/{}/{}/{}.txt '.format(args.dataset, args.method.lower(), emb_path) + \
                  '--method {} --task link-prediction '.format(args.method) + \
                  '--dimensions {} '.format(args.emb_size) + \
                  ('--weighted True' if args.weighted == 'True' else '')
                  #'--number-walks {} --walk-length {} --window-size 10 '.format(args.num_walks, args.walk_len) + \
                  #'--p {} --q {} '.format(args.p, args.q) + \

        print(command)
        os.system(command)
        emb_dict = load_embedding('./embeddings/{}/{}/{}.txt'.format(args.dataset, args.method.lower(), emb_path))

        if args.save_emb:
            emb_dict = load_embedding('./embeddings/{}/{}/{}.txt'.format(args.dataset, args.method.lower(), emb_path))
            emb = np.zeros((len(emb_dict.keys()), args.emb_size))
            for i in range(emb.shape[0]):
                emb[i, :] = emb_dict[str(i)]

            if args.all_regions:
                np.save('./embeddings/{}/{}/{}_full.npy'.format(args.dataset, args.method.lower(), emb_path), emb)
                # ToDo: better removal of _90 in interactions
                gene_idxs = np.load(
                    '/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/gene_idxs/{}.npy'.format(
                        args.dataset, args.interactions[:-3]))
                emb = emb[gene_idxs]

            np.save('./embeddings/{}/{}/{}_prova.npy'.format(args.dataset, args.method.lower(), emb_path), emb)
        os.remove('./embeddings/{}/{}/{}.txt'.format(args.dataset, args.method.lower(), emb_path))
        os.remove('data/{}/interactions/{}.edgelist'.format(args.dataset, args.interactions))
