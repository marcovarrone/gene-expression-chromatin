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
    #parser.add_argument('--interactions', type=str, default='primary_observed_KR_all_50000_50000_primary_observed_NONE_all_50000_50000')
    parser.add_argument('--interactions', type=str, default='primary_observed_KR_1_1_40000_40000_2.42')
    parser.add_argument('--emb-size', type=int, default=8)
    parser.add_argument('--n-jobs', type=int, default=10)
    parser.add_argument('--all-regions', default=False, action='store_true')
    parser.add_argument('--chr-gene-idxs', type=int, default=None)  # ToDo: remove
    parser.add_argument('--genes-chr', default=False, action='store_true')
    parser.add_argument('--force', default=False, action='store_true')
    parser.add_argument('--method', default='SVD', type=str, choices=['Laplacian', 'SVD', 'HOPE'])
    parser.add_argument('--weighted', type=str, default='False')
    parser.add_argument('--save-emb', default=False, action='store_true')
    parser.add_argument('--task', type=str, default='link-prediction')
    args = parser.parse_args()

    set_n_threads(args.n_jobs)

    interactions_path = './data/{}/interactions/interactions_{}.npy'.format(
        args.dataset, args.interactions)

    emb_path = '{}_es{}'.format(args.interactions, args.emb_size)
    if not os.path.exists('embeddings/{}/{}/{}.npy'.format(args.dataset, args.method, emb_path)) or args.force:
        adj = np.load(interactions_path)
        adj[np.isnan(adj)] = 0
        if adj.shape[0] != adj.shape[1]:
            # adj = np.block([[np.zeros((adj.shape[0], adj.shape[0])), adj], [adj.T, np.zeros((adj.shape[1], adj.shape[1]))]])
            graph = nx.algorithms.bipartite.from_biadjacency_matrix(sps.csr_matrix(adj))
        else:
            graph = nx.from_numpy_array(adj)

        if args.weighted == 'True':
            nx.write_weighted_edgelist(graph,
                                       'data/{}/interactions/{}.edgelist'.format(args.dataset, args.interactions))
        else:
            nx.write_edgelist(graph, 'data/{}/interactions/{}.edgelist'.format(args.dataset, args.interactions), data=False)

        command = 'bionev --input data/{}/interactions/{}.edgelist '.format(args.dataset, args.interactions) + \
                  '--output ./embeddings/{}/{}/{}.txt '.format(args.dataset, args.method.lower(), emb_path) + \
                  '--method {} --task {} '.format(args.method, args.task) + \
                  '--dimensions {} '.format(args.emb_size) + \
                  ('--weighted True' if args.weighted == 'True' else '')
        print(command)
        os.system(command)
        emb_dict = load_embedding('./embeddings/{}/{}/{}.txt'.format(args.dataset, args.method.lower(), emb_path))

        if args.save_emb:
            emb_dict = load_embedding('./embeddings/{}/{}/{}.txt'.format(args.dataset, args.method.lower(), emb_path))

            genes = range(len(emb_dict.keys()))
            if args.genes_chr:
                genes_chr = np.load(
                    '/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/genes_chr/{}.npy'.format(
                        args.dataset, args.interactions))
                genes = np.where(genes_chr != 0)[0]

            emb = np.zeros((len(genes), args.emb_size))

            for i, gene in enumerate(genes):
                try:
                    emb[i, :] = emb_dict[str(gene)]
                except KeyError:
                    np.delete(emb, i, axis=0)
                    if args.genes_chr:
                        genes_chr[gene] = 0
                        np.save('/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/genes_chr/{}.npy'.format(
                        args.dataset, args.interactions), genes_chr)

            if args.all_regions:
                np.save('./embeddings/{}/{}/{}_full.npy'.format(args.dataset, args.method.lower(), emb_path), emb)
                # ToDo: better removal of _90 in interactions
                gene_idxs = np.load(
                    '/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/gene_idxs/{}.npy'.format(
                        args.dataset, args.interactions[:-3]))
                emb = emb[gene_idxs]

            if args.chr_gene_idxs:
                gene_idxs = np.load(
                    '/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/gene_idxs/{}.npy'.format(
                        args.dataset, args.interactions))
                emb = emb[gene_idxs[args.chr_gene_idxs]]

            np.save('./embeddings/{}/{}/{}.npy'.format(args.dataset, args.method.lower(), emb_path), emb)
        os.remove('./embeddings/{}/{}/{}.txt'.format(args.dataset, args.method.lower(), emb_path))
        os.remove('data/{}/interactions/{}.edgelist'.format(args.dataset, args.interactions))