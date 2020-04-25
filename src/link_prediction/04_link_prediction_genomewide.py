import argparse
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from utils_link_prediction import *
from utils import set_n_threads, set_gpu

def main(args):
    if not args.chromatin_network_name:
        args.chromatin_network_name = '{}_{}_{}_'.format(args.type, 'all', args.bin_size) + \
                                      '{}_{}'.format(args.hic_threshold_intra, args.hic_threshold_inter)
    #args.chromatin_network_name += '_w{:.1f}'.format(args.weight)

    args.coexp_thr = '{}_{}'.format(args.coexp_thr_intra, args.coexp_thr_inter)
    args, filename = setup_filenames_and_folders(args, 'all')

    if not os.path.exists('../../results/{}/{}'.format(args.dataset, filename)) or args.force:
        coexpression = load_coexpression(args, args.chromatin_network_name, 'all')

        mask = get_mask_intra(args.dataset)
        edges_intra, non_edges_intra = get_edges(coexpression * mask)


        mask_inter = mask.copy()
        opposite_idxs = np.isnan(mask_inter)
        mask_inter[mask_inter == 1] = np.nan
        mask_inter[opposite_idxs] = 1

        edges_inter, non_edges_inter = get_edges(coexpression * mask_inter, n_eges_intra=edges_intra.shape[0],
                                                 inter_ratio=args.inter_ratio)


        edges = np.vstack((edges_intra, edges_inter))
        non_edges = np.vstack((non_edges_intra, non_edges_inter))

        X_train, X_test, y_train, y_test = build_dataset(args, edges, non_edges, coexpression.shape[0])

        link_prediction(args, X_train, y_train, X_test, y_test, filename)
    else:
        print('Result already computed for {}. Skipped.'.format(filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, required=True)
    parser.add_argument('--n-iter', type=int, default=1)
    parser.add_argument('--cv-splits', type=int, default=5)
    parser.add_argument('--method', type=str, default='node2vec',
                        choices=['random', 'distance', 'topological', 'svd', 'node2vec'])

    parser.add_argument('--chromatin-network-name', type=str, default=None)
    parser.add_argument('--type', type=str)
    parser.add_argument('--bin-size', type=int)
    parser.add_argument('--hic-threshold-intra', type=str)
    parser.add_argument('--hic-threshold-inter', type=str)
    #parser.add_argument('--weight', type=float, default=1.0)

    parser.add_argument('--aggregators', nargs='*', default=['hadamard'])
    parser.add_argument('--classifier', default='rf', choices=['mlp', 'lr', 'svm', 'rf'])
    parser.add_argument('--coexp-thr-intra', type=str, default=None, required=True)
    parser.add_argument('--coexp-thr-inter', type=str, default=None, required=True)
    parser.add_argument('--save-predictions', default=True, action='store_true')
    parser.add_argument('--emb-size', type=int, default=16)
    parser.add_argument('--inter-ratio', type=float, default=0.1)

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
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--project', default='n2v_hic_tuning', type=str)

    args = parser.parse_args()

    seed = 42
    np.random.seed(seed)

    set_gpu(args.gpu)
    set_n_threads(args.n_jobs)

    main(args)
