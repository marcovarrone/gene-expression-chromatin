import argparse
import pickle
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.model_selection import train_test_split

from utils_link_prediction import *
from utils import set_n_threads, set_gpu


def main(args):
    args.embedding = ''
    if args.method != 'distance':
        args.embedding = 'es{}'.format(args.emb_size)

    if args.method == 'node2vec':
        args.embedding += '_nw{}_wl{}_p{}_q{}'.format(args.num_walks, args.walk_len, args.p, args.q)

    if args.chr_src is None:
        raise ValueError()

    if args.chr_tgt is None:
        args.chr_tgt = args.chr_src

    args.interactions = '{}_{}_{}_{}_{}_{}_{}'.format(args.file, args.type, args.norm, args.chr_src, args.chr_tgt,
                                                      args.bin_size, args.hic_thr)

    args.aggregators = '_'.join(args.aggregators)

    args.folder = 'chromatin_networks'
    args.name = args.interactions

    args.embedding = args.name + '_' + args.embedding

    os.makedirs('../../results/{}/chr_{}'.format(args.dataset, args.chr_src), exist_ok=True)
    os.makedirs('../../results/{}/predictions/chr_{}'.format(args.dataset, args.chr_src), exist_ok=True)

    if args.method == 'topological':
        filename = '{}chr_{}/{}_{}_{}_{}.pkl'.format('test/' if args.test else '', args.chr_src, args.classifier,
                                                     args.method, args.name, args.aggregators)

    else:
        filename = '{}chr_{}/{}_{}_{}_{}_{}.pkl'.format('test/' if args.test else '', args.chr_src, args.classifier,
                                                        args.method, args.embedding, args.aggregators, args.coexp_thr)

    if not os.path.exists('../../results/{}/{}'.format(args.dataset, filename)) or args.force:
        coexpression = np.load(
            '../../data/{}/coexpression_networks/coexpression_chr_{}_{}_{}.npy'.format(args.dataset, args.chr_src,
                                                                                       args.chr_tgt, args.coexp_thr))

        disconnected_nodes = np.load(
            '../../data/{}/disconnected_nodes/{}.npy'.format(
                args.dataset, args.name))

        print("N. disconnected nodes:", len(disconnected_nodes))
        if len(disconnected_nodes) > 0:
            coexpression[disconnected_nodes] = np.nan
            coexpression[:, disconnected_nodes] = np.nan

        n_nodes = coexpression.shape[0]

        coexpression_intra = coexpression

        edges_intra = np.array(np.argwhere(coexpression_intra == 1))
        edges_intra_nodes = np.unique(edges_intra)

        non_nodes_intra = np.setdiff1d(np.arange(n_nodes), edges_intra_nodes)

        coexpression_intra_neg = coexpression_intra.copy()
        coexpression_intra_neg[non_nodes_intra, :] = np.nan
        coexpression_intra_neg[:, non_nodes_intra] = np.nan

        non_edges_intra = np.array(np.argwhere(coexpression_intra_neg == 0))
        non_edges_intra = non_edges_intra[
            np.random.choice(non_edges_intra.shape[0], edges_intra.shape[0], replace=False)]

        edges = edges_intra
        non_edges = non_edges_intra

        n_edges = edges.shape[0]
        n_non_edges = non_edges.shape[0]

        if args.method == 'topological':
            X = topological_features(args, edges, non_edges)
        elif args.method == 'ids':
            X = np.vstack((edges, non_edges))
        elif args.method == 'distance':
            X = distance_embedding(args.dataset, edges, non_edges, chr_src=args.chr_src)
        else:
            X = method_embedding(args, n_nodes, edges, non_edges)
        y = np.hstack((np.ones(n_edges), np.zeros(n_non_edges)))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

        results = defaultdict(list)
        if args.test:
            results = evaluate_embedding(X_train, y_train, args.classifier, verbose=1, clf_params={'n_estimators': 100},
                                         X_test=X_test, y_test=y_test)
        else:
            for i in range(args.n_iter):
                results_iter = evaluate_embedding(X_train, y_train, args.classifier, verbose=1,
                                                  clf_params={'n_estimators': 100}, cv_splits=args.cv_splits)
                for key in results_iter.keys():
                    results[key].extend(results_iter[key])

        with open('../../results/{}/{}'.format(args.dataset, filename), 'wb') as file_save:
            pickle.dump(results, file_save)

        print("Mean Accuracy:", np.mean(results['acc']), "- Mean ROC:", np.mean(results['roc']), "- Mean F1:",
              np.mean(results['f1']),
              "- Mean Precision:", np.mean(results['precision']), "- Mean Recall", np.mean(results['recall']))
    else:
        print('Result already computed for {}. Skipped.'.format(filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--chr-src', type=int, required=True)
    parser.add_argument('--chr-tgt', type=int, default=None)
    parser.add_argument('--n-iter', type=int, default=1)
    parser.add_argument('--cv-splits', type=int, default=5)
    parser.add_argument('--method', type=str, default='node2vec',
                        choices=['distance', 'topological', 'svd', 'node2vec'])

    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--norm', type=str, required=True)
    parser.add_argument('--bin-size', type=int, required=True)
    parser.add_argument('--hic-thr', type=str, required=True)

    parser.add_argument('--aggregators', nargs='*', default=['hadamard'])
    parser.add_argument('--classifier', default='rf', choices=['mlp', 'lr', 'svm', 'rf'])
    parser.add_argument('--coexp-thr', type=str, required=True)
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

    args = parser.parse_args()

    seed = 42
    np.random.seed(seed)

    set_gpu(args.gpu)
    set_n_threads(args.n_jobs)

    main(args)
