import argparse
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from utils_link_prediction import *
from utils import set_n_threads, set_gpu


def main(args):
    if args.chr_src is None:
        raise ValueError()

    if args.chr_tgt is None:
        args.chr_tgt = args.chr_src

    if args.chromatin_network_name is None:
        args.chromatin_network_name = '{}_{}_{}_{}_{}'.format(args.type, args.chr_src, args.chr_tgt, args.bin_size, args.hic_threshold)

    args, filename = setup_filenames_and_folders(args, args.chr_src)

    if not os.path.exists('../../results/{}/{}'.format(args.dataset, filename)) or args.force:
        coexpression = load_coexpression(args, args.chromatin_network_name, '{}_{}'.format(args.chr_src, args.chr_tgt))

        edges, non_edges = get_edges(coexpression)

        X_train, X_test, y_train, y_test = build_dataset(args, edges, non_edges, coexpression.shape[0])
        print(X_train.shape)

        link_prediction(args, X_train, y_train, X_test, y_test, filename)
    else:
        # ToDo: load results and print them
        print('Result already computed for {}. Skipped.'.format(filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--chr-src', type=int, default=1)
    parser.add_argument('--chr-tgt', type=int, default=1)
    parser.add_argument('--n-iter', type=int, default=1)
    parser.add_argument('--cv-splits', type=int, default=5)
    parser.add_argument('--method', type=str, default='node2vec',
                        choices=['random', 'distance', 'topological', 'svd', 'node2vec'])

    parser.add_argument('--type', type=str)
    parser.add_argument('--bin-size', type=int)
    parser.add_argument('--hic-threshold', type=str)
    parser.add_argument('--chromatin-network-name', type=str)

    parser.add_argument('--aggregators', nargs='*', default=['hadamard'])
    parser.add_argument('--classifier', default='rf', choices=['mlp', 'lr', 'svm', 'rf', 'random'])
    parser.add_argument('--coexp-thr', type=str, default=None, required=True)
    parser.add_argument('--save-predictions', default=True, action='store_true')
    parser.add_argument('--emb-size', type=int, default=16)

    # Topological measures params
    parser.add_argument('--edge-features', default=True, action='store_true')

    # Node2vec params
    parser.add_argument('--num-walks', type=int, default=10)
    parser.add_argument('--walk-len', type=int, default=80)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--q', type=float, default=1.0)
    parser.add_argument('--window', type=int, default=10)

    parser.add_argument('--force', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--n-jobs', type=int, default=10)
    parser.add_argument('--wandb', default=False, action='store_false')
    parser.add_argument('--project', default='parameter-importance', type=str)
    args = parser.parse_args()

    seed = 42
    np.random.seed(seed)

    set_gpu(args.gpu)
    set_n_threads(args.n_jobs)

    main(args)
