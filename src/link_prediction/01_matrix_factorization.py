import argparse
import os

from utils import set_n_threads
from utils_link_prediction import generate_embedding

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--chromatin-network', type=str, required=True)
    parser.add_argument('--emb-size', type=int, default=16)
    parser.add_argument('--n-jobs', type=int, default=10)
    parser.add_argument('--force', default=False, action='store_true')
    parser.add_argument('--method', default='SVD', type=str, choices=['Laplacian', 'SVD', 'HOPE'])
    parser.add_argument('--weighted', type=str, default='False')
    parser.add_argument('--save-emb', default=False, action='store_true')
    parser.add_argument('--task', type=str, default='link-prediction', choices=['none', 'link-prediction'])
    args = parser.parse_args()

    set_n_threads(args.n_jobs)

    args.folder = 'chromatin_networks'
    args.name = args.chromatin_network

    interactions_path = '../../data/{}/chromatin_networks/{}.npy'.format(args.dataset, args.name)

    emb_path = '{}_es{}'.format(args.chromatin_network, args.emb_size)


    if not os.path.exists(
            '../../data/{}/embeddings/{}/{}.npy'.format(args.dataset, args.method.lower(), emb_path)) or args.force:
        command = 'bionev --input ../../data/{}/chromatin_networks/{}.edgelist '.format(args.dataset,
                                                                                        args.chromatin_network) + \
                  '--output ../../data/{}/embeddings/{}/{}.txt '.format(args.dataset, args.method.lower(), emb_path) + \
                  '--method {} --task {} '.format(args.method, args.task) + \
                  '--dimensions {} '.format(args.emb_size) + \
                  '--weighted True' if args.weighted else ''

        generate_embedding(args, emb_path, interactions_path, command)
    else:
        print(print('Embeddings already computed for {}. Skipped.'.format('../../data/{}/embeddings/{}/{}.npy'.format(args.dataset, args.method.lower(), emb_path))))
