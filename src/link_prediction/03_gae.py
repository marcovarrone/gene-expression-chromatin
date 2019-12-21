import argparse
import os

import networkx as nx
import numpy as np
from bionev.utils import load_embedding

from link_prediction.utils import set_gpu, set_n_threads

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MCF7')
    parser.add_argument('--interactions', type=str, default='primary_observed_KR_1_1_50000_50000_0.9073')
    parser.add_argument('--emb-size', type=int, default=8)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--n-jobs', type=int, default=10)
    parser.add_argument('--all-regions', default=False, action='store_true')
    parser.add_argument('--chr', type=int, default=1)
    parser.add_argument('--genes-chr', default=False, action='store_true')
    parser.add_argument('--force', default=False, action='store_true')
    parser.add_argument('--method', default='gae', choices=['gae', 'gvae'])
    parser.add_argument('--weighted', type=str, default='False')
    parser.add_argument('--save-emb', default=False, action='store_true')
    args = parser.parse_args()

    set_gpu(args.gpu)
    set_n_threads(args.n_jobs)

    interactions_path = './data/{}/interactions/interactions_{}.npy'.format(
        args.dataset, args.interactions)

    emb_path = '{}_es{}_h{}_e{}_lr{}_do{}'.format(args.interactions, args.emb_size, args.hidden, args.epochs,
                                                  args.lr, args.dropout)
    if not os.path.exists('embeddings/{}/{}/{}.npy'.format(args.dataset, args.method, emb_path)) or args.force:
        adj = np.load(interactions_path)
        adj[np.isnan(adj)] = 0
        graph = nx.from_numpy_array(adj)

        if args.weighted == 'True':
            nx.write_weighted_edgelist(graph,
                                       'data/{}/interactions/{}.edgelist'.format(args.dataset, args.interactions))
        else:
            nx.write_edgelist(graph, 'data/{}/interactions/{}.edgelist'.format(args.dataset, args.interactions))

        # ToDo: add weighted parameter
        command = 'bionev --input data/{}/interactions/{}.edgelist '.format(args.dataset, args.interactions) + \
                  '--output ./embeddings/{}/{}/{}.txt '.format(args.dataset, args.method, emb_path) + \
                  '--method GAE --task link-prediction ' + \
                  '--gae_model_selection {} '.format('gcn_ae' if args.method == 'gae' else 'gcn_vae') + \
                  '--dimensions {} '.format(args.emb_size) + \
                  '--epochs {} '.format(args.epochs) + \
                  '--hidden {} '.format(args.hidden) + \
                  '--lr {} '.format(args.lr) + \
                  '--dropout {} '.format(args.dropout) + \
                  ('--weighted True' if args.weighted == 'True' else '')
        print(command)
        os.system(command)

        if args.save_emb:
            emb_dict = load_embedding('./embeddings/{}/{}/{}.txt'.format(args.dataset, args.method, emb_path))
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
                        np.save(
                            '/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/genes_chr/{}.npy'.format(
                                args.dataset, args.interactions), genes_chr)

            if args.all_regions:
                np.save('./embeddings/{}/{}/{}_full.npy'.format(args.dataset, args.method, emb_path), emb)
                # ToDo: better removal of _90 in interactions
                gene_idxs = np.load(
                    '/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/gene_idxs/{}.npy'.format(
                        args.dataset, args.interactions[:-3]))
                emb = emb[gene_idxs]


            np.save('./embeddings/{}/{}/{}.npy'.format(args.dataset, args.method, emb_path), emb)
        os.remove('./embeddings/{}/{}/{}.txt'.format(args.dataset, args.method, emb_path))
        os.remove('data/{}/interactions/{}.edgelist'.format(args.dataset, args.interactions))
