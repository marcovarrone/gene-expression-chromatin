import argparse
import os

import networkx as nx
import numpy as np
from bionev.utils import load_embedding
from link_prediction.utils import set_gpu, set_n_threads
import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='GM19238')
    parser.add_argument('--interactions', type=str, default='combined_oe_KR_1_1_10000_90')
    parser.add_argument('--emb-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--n-jobs', type=int, default=10)
    parser.add_argument('--all-regions', default=False, action='store_true')
    parser.add_argument('--chr', type=int, default=1)
    parser.add_argument('--force', default=False, action='store_true')
    parser.add_argument('--method', default='line')
    parser.add_argument('--save-emb', default=False, action='store_true')
    args = parser.parse_args()

    set_gpu(args.gpu)
    set_n_threads(args.n_jobs)

    interactions_path = './data/{}/interactions/interactions_{}.npy'.format(
        args.dataset, args.interactions)

    emb_path = '{}_es{}_e{}_lr{}'.format(args.interactions, args.emb_size, args.epochs,
                                                  args.lr)
    if not os.path.exists('embeddings/{}/{}.npy'.format(args.method, emb_path)) or args.force:
        graph = nx.from_numpy_array(np.load(interactions_path))

        nx.write_edgelist(graph, 'data/{}/interactions/{}.edgelist'.format(args.dataset, args.interactions))

        #ToDo: add weighted parameter
        command = 'bionev --input data/{}/interactions/{}.edgelist '.format(args.dataset, args.interactions) + \
                  '--output ./embeddings/{}/{}.txt '.format(args.method,emb_path) + \
                  '--method LINE --task link-prediction ' + \
                  '--dimensions {} '.format(args.emb_size) + \
                  '--epochs {} '.format(args.epochs) + \
                  '--lr {} '.format(args.lr)
        print(command)
        os.system(command)

        if args.save_emb:
            emb_dict = load_embedding('./embeddings/{}/{}.txt'.format(args.method, emb_path))
            emb = np.zeros((len(emb_dict.keys()), args.emb_size))
            for i in range(emb.shape[0]):
                emb[i, :] = emb_dict[str(i)]

            if args.all_regions:
                np.save('./embeddings/{}/{}_full.npy'.format(args.method, emb_path), emb)
                # ToDo: better removal of _90 in interactions
                gene_idxs = np.load(
                    '/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/gene_idxs/{}.npy'.format(
                        args.dataset, args.interactions[:-3]))
                emb = emb[gene_idxs]

            np.save('./embeddings/{}/{}.npy'.format(args.method, emb_path), emb)
        os.remove('./embeddings/{}/{}.txt'.format(args.method, emb_path))
        os.remove('data/{}/interactions/{}.edgelist'.format(args.dataset, args.interactions))
