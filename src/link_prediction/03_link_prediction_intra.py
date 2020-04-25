import argparse
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from utils_link_prediction import *
from utils import set_n_threads, set_gpu
import matplotlib.pyplot as plt


def main(args):
    args.chromatin_network_name = '{}_all_{}_{}'.format(args.type,args.bin_size, args.hic_threshold)

    args, filename = setup_filenames_and_folders(args, 'all')

    edges = None
    non_edges = None
    offset = 0
    if not os.path.exists('../../results/{}/{}'.format(args.dataset, filename)) or args.force:
        for i in range(1, 23):
            chromatin_network_chr_name = '{}_{}_{}_{}_{}'.format(args.type, i, i,
                                                                       args.bin_size,
                                                                       args.hic_threshold)

            coexpression = load_coexpression(args, chromatin_network_chr_name, '{}_{}'.format(i, i))

            edges_intra, non_edges_intra = get_edges(coexpression)

            edges_intra += offset
            non_edges_intra += offset

            if edges is None and non_edges is None:
                edges = edges_intra
                non_edges = non_edges_intra
            else:
                edges = np.vstack((edges, edges_intra))
                non_edges = np.vstack((non_edges, non_edges_intra))

            offset += coexpression.shape[0]

        adj = np.zeros((offset, offset))
        adj[edges[:, 0], edges[:, 1]] = 1
        plt.figure(dpi=500)
        plt.imshow(adj, cmap='Reds')
        plt.show()

        adj = np.zeros((offset, offset))
        adj[non_edges[:, 0], non_edges[:, 1]] = 1
        plt.figure(dpi=500)
        plt.imshow(adj, cmap='Reds')
        plt.show()

        X_train, X_test, y_train, y_test = build_dataset(args, edges, non_edges, offset)
        print(X_train.shape)

        link_prediction(args, X_train, y_train, X_test, y_test, filename)
    else:
        # ToDo: load results and print them
        print('Result already computed for {}. Skipped.'.format(filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--n-iter', type=int, default=1)
    parser.add_argument('--cv-splits', type=int, default=5)
    parser.add_argument('--method', type=str, default='node2vec',
                        choices=['random', 'distance', 'topological', 'svd', 'node2vec'])

    parser.add_argument('--type', type=str, default='observed')
    parser.add_argument('--bin-size', type=int, default=40000)
    parser.add_argument('--hic-threshold', type=str, required=True)

    parser.add_argument('--edge-features', default=True, action='store_true')
    parser.add_argument('--aggregators', nargs='*', default=['hadamard'])
    parser.add_argument('--classifier', default='rf', choices=['mlp', 'lr', 'svm', 'mlp_2', 'rf'])
    parser.add_argument('--coexp-thr', type=str, required=True)
    parser.add_argument('--save-predictions', default=False, action='store_true')
    parser.add_argument('--emb-size', type=int, default=16)
    parser.add_argument('--force', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')

    # Node2vec params
    parser.add_argument('--num-walks', type=int, default=10)
    parser.add_argument('--walk-len', type=int, default=80)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--q', type=float, default=1.0)
    parser.add_argument('--window', type=int, default=10)

    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--n-jobs', type=int, default=10)
    parser.add_argument('--wandb', default=True, action='store_true')
    parser.add_argument('--project', type=str, default='n2v_hic_tuning')

    args = parser.parse_args()

    seed = 42
    np.random.seed(seed)

    set_gpu(args.gpu)
    set_n_threads(args.n_jobs)

    main(args)
