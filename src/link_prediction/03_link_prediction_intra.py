import argparse
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from utils_link_prediction import *
from utils import set_n_threads, set_gpu


def main(args):
    args.chromatin_network_name = '{}_{}_{}_all_{}_{}'.format(args.file, args.type, args.norm,
                                                             args.bin_size, args.hic_threshold)

    args, filename = setup_filenames_and_folders(args, 'all')

    edges = None
    non_edges = None
    offset = 0
    if not os.path.exists('../../results/{}/{}'.format(args.dataset, filename)) or args.force:
        for i in range(1, 23):
            chromatin_network_chr_name = '{}_{}_{}_{}_{}_{}_{}'.format(args.file, args.type, args.norm, i, i, args.bin_size,
                                                         args.hic_threshold)

            coexpression = load_coexpression(args, i, i, chromatin_network_chr_name)

            edges_intra, non_edges_intra = get_edges_intra(coexpression)

            edges_intra += offset
            non_edges_intra += offset

            if edges is None and non_edges is None:
                edges = edges_intra
                non_edges = non_edges_intra
            else:
                edges = np.vstack((edges, edges_intra))
                non_edges = np.vstack((non_edges, non_edges_intra))

            offset += coexpression.shape[0]

        X_train, X_test, y_train, y_test = build_dataset(args, edges, non_edges, offset)
        print(X_train.shape)

        link_prediction(args, X_train, y_train, X_test, y_test, filename)
    else:
        #ToDo: load results and print them
        print('Result already computed for {}. Skipped.'.format(filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='prostate')
    parser.add_argument('--n-iter', type=int, default=1)
    parser.add_argument('--cv-splits', type=int, default=5)
    parser.add_argument('--method', type=str, default='topological',
                        choices=['distance', 'topological', 'svd', 'node2vec'])

    parser.add_argument('--file-intra', type=str, required=True)
    parser.add_argument('--type-intra', type=str, required=True)
    parser.add_argument('--norm-intra', type=str, required=True)
    parser.add_argument('--bin-size-intra', type=int, required=True)
    parser.add_argument('--hic-threshold-intra', type=str, required=True)

    parser.add_argument('--file-intra', type=str, required=True)
    parser.add_argument('--type-intra', type=str, required=True)
    parser.add_argument('--norm-intra', type=str, required=True)
    parser.add_argument('--bin-size-intra', type=int, required=True)
    parser.add_argument('--hic-threshold-intra', type=str, required=True)

    parser.add_argument('--edge-features', default=True, action='store_true')
    parser.add_argument('--aggregators', nargs='*', default=['avg', 'l1'])
    parser.add_argument('--classifier', default='rf', choices=['mlp', 'lr', 'svm', 'mlp_2', 'rf'])
    parser.add_argument('--coexp-thr', type=str, default='0.57')
    parser.add_argument('--save-predictions', default=True, action='store_true')
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

    args = parser.parse_args()

    seed = 42
    np.random.seed(seed)

    set_gpu(args.gpu)
    set_n_threads(args.n_jobs)

    main(args)
