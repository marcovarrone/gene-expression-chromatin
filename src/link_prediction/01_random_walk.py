import argparse

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from utils import set_n_threads
from utils_link_prediction import generate_embedding

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='prostate')
    parser.add_argument('--chromatin-network', type=str,
                        default='primary_observed_ICE_all_40000_3.59')
    parser.add_argument('--coexpression', type=str, default=None)
    parser.add_argument('--emb-size', type=int, default=16)
    parser.add_argument('--n-jobs', type=int, default=10)
    parser.add_argument('--num-walks', type=int, default=10)
    parser.add_argument('--walk-len', type=int, default=80)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--q', type=float, default=1.0)
    parser.add_argument('--window', type=int, default=10)

    parser.add_argument('--force', default=True, action='store_true')
    parser.add_argument('--method', default='node2vec', type=str, choices=['node2vec', 'struc2vec'])
    parser.add_argument('--weighted', type=str, default='False')
    parser.add_argument('--save-emb', default=False, action='store_true')
    parser.add_argument('--task', type=str, default='none', choices=['none', 'link-prediction'])

    args = parser.parse_args()

    set_n_threads(args.n_jobs)

    if args.coexpression:
        args.folder = 'coexpression'
        args.name = args.coexpression
    else:
        args.folder = 'chromatin_networks'
        args.name = args.chromatin_network

    interactions_path = '../../data/{}/{}/{}.npy'.format(
        args.dataset, args.folder, args.name)

    emb_path = '{}_es{}_nw{}_wl{}_p{}_q{}'.format(args.name, args.emb_size,
                                                    args.num_walks, args.walk_len, args.p, args.q)

    command = 'bionev --input ../../data/{}/chromatin_networks/{}.edgelist '.format(args.dataset, args.name) + \
              '--output ../../data/{}/embeddings/{}/{}.txt '.format(args.dataset, args.method.lower(), emb_path) + \
              '--method {} --task {} '.format(args.method, args.task) + \
              '--dimensions {} '.format(args.emb_size) + \
              '--number-walks {} '.format(args.num_walks) + \
              '--walk-length {} '.format(args.walk_len) + \
              '--p {} --q {} '.format(args.p, args.q) + \
              '--window-size {} '.format(args.window)# + \
              #'--weighted False'

    generate_embedding(args, emb_path, interactions_path, command)