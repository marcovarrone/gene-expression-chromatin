import argparse
import hashlib
import pickle
import warnings
import wandb

warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.model_selection import train_test_split

from utils_link_prediction import *
from utils import set_n_threads, set_gpu, intra_mask

def main(args):
    args.embedding = ''
    if args.method != 'distance':
        args.embedding = 'es{}'.format(args.emb_size)

    if args.method == 'node2vec':
        args.embedding += '_nw{}_wl{}_p{}_q{}'.format(args.num_walks, args.walk_len, args.p, args.q)

    coexp_thrs = '_'.join(args.coexp_thrs)

    hic_files = ['{}_{}_{}'.format(file, type, norm) for
                 file, type, norm in zip(args.files, args.types, args.norms)]

    hic_preprocessings = ['{}_{}{}'.format(window, threshold, ('_' + str(weight)) if weight else '') for
                          window, threshold, weight in
                          zip(args.windows, args.hic_thrs, args.weights)]

    args.interactions = ['{}_all_{}'.format(file, preprocessing) for
                         file, preprocessing in zip(hic_files, hic_preprocessings)]

    args.interactions = '_'.join(args.interactions)

    args.aggregators = '_'.join(args.aggregators)

    if args.coexp_features:
        args.folder = 'coexpression_networks'
        args.name = 'chr_all_{}'.format(coexp_thrs)
    else:
        args.folder = 'chromatin_networks'
        args.name = args.interactions

    args.embedding = args.name + '_' + args.embedding

    os.makedirs('../../results/{}/chr_all'.format(args.dataset), exist_ok=True)
    os.makedirs('../../results/{}/predictions/chr_all'.format(args.dataset), exist_ok=True)

    if args.method == 'topological':
        filename = '{}chr_all/{}_{}_{}_{}.pkl'.format('test/' if args.test else '', args.classifier,
                                                      args.method, args.name, args.aggregators)
    else:
        filename = '{}chr_all/{}_{}_{}_{}_{}.pkl'.format('test/' if args.test else '', args.classifier, args.method,
                                                         args.embedding, args.aggregators, coexp_thrs)

    edges = None
    non_edges = None
    offset = 0
    for i in range(1, 23):
        args.interactions = ['{}_{}_{}_{}'.format(file,i, i, preprocessing) for
                             file, preprocessing in zip(hic_files, hic_preprocessings)]

        args.interactions = '_'.join(args.interactions)

        if args.coexp_features:
            args.folder = 'coexpression_networks'
            args.name = 'chr_{}_{}_{}'.format(i, i, coexp_thrs)
        else:
            args.folder = 'chromatin_networks'
            args.name = args.interactions

        coexpression = np.load(
            '../../data/{}/coexpression_networks/coexpression_chr_{}_{}_{}.npy'.format(args.dataset, i,
                                                                                       i, coexp_thrs))
        print(np.sum(coexpression ==1))

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

        edges_intra += offset
        non_edges_intra += offset

        if edges is None and non_edges is None:
            edges = edges_intra
            non_edges = non_edges_intra
        else:
            edges = np.vstack((edges, edges_intra))
            non_edges = np.vstack((non_edges, non_edges_intra))
            print(edges.shape, non_edges.shape)

        n_edges = edges.shape[0]
        n_non_edges = non_edges.shape[0]

        offset += coexpression.shape[0]


    if args.method == 'topological':
        X = topological_features(args, edges, non_edges)
    elif args.method == 'ids':
        X = np.vstack((edges, non_edges))
    elif args.method == 'distance':
        X = distance_embedding(args.dataset, edges, non_edges)
    else:
        X = method_embedding(args, n_nodes, edges, non_edges)
    y = np.hstack((np.ones(n_edges), np.zeros(n_non_edges)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    print(X_train.shape)

    results = defaultdict(list)
    if args.test:
        results = evaluate_embedding(X_train, y_train, args.classifier, verbose=1, clf_params={'n_estimators': 100},
                                     X_test=X_test, y_test=y_test)
        if args.wandb:
            wandb.run.summary.update({'accuracy': results['acc'],
                                      'accuracy std': results['acc'],
                                      'precision': results['precision'],
                                      'precision std': results['precision'],
                                      'recall': results['recall'],
                                      'recall std': results['recall']})
    else:
        for i in range(args.n_iter):
            results_iter = evaluate_embedding(X_train, y_train, args.classifier, verbose=1,
                                              clf_params={'n_estimators': 100}, cv_splits=args.cv_splits)
            for key in results_iter.keys():
                results[key].extend(results_iter[key])

        if args.wandb:
            for _ in results.keys():
                wandb.run.summary.update({'accuracy': np.mean(results['acc']),
                                          'accuracy std': np.std(results['acc']),
                                          'precision': np.mean(results['precision']),
                                          'precision std': np.std(results['precision']),
                                          'recall': np.mean(results['recall']),
                                          'recall std': np.std(results['recall'])})

    with open('../../results/{}/{}'.format(args.dataset, filename), 'wb') as file_save:
        pickle.dump(results, file_save)

    if args.wandb:
        wandb.save('../../results/{}/{}'.format(args.dataset, filename))

    print("Mean Accuracy:", np.mean(results['acc']), "- Mean ROC:", np.mean(results['roc']), "- Mean F1:",
          np.mean(results['f1']),
          "- Mean Precision:", np.mean(results['precision']), "- Mean Recall", np.mean(results['recall']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='prostate')
    parser.add_argument('--n-iter', type=int, default=1)
    parser.add_argument('--cv-splits', type=int, default=5)
    parser.add_argument('--method', type=str, default='node2vec',
                        choices=['distance', 'topological', 'svd', 'node2vec'])

    parser.add_argument('--windows', type=int, nargs='*', default=[40000, 40000])
    parser.add_argument('--hic-thrs', type=str, nargs='*', default=['3.64', '1.23'])
    parser.add_argument('--weights', type=str, nargs='*', default=[None])
    parser.add_argument('--norms', nargs='*', type=str, choices=['NONE', 'VC', 'VC_SQRT', 'KR', 'ICE'],
                        default=['KR', 'KR'])
    parser.add_argument('--types', nargs='*', type=str, choices=['observed', 'observed'], default=['observed', 'observed'])
    parser.add_argument('--files', nargs='*', type=str, default=['primary', 'primary'])

    parser.add_argument('--coexp-features', default=False, action='store_true')
    parser.add_argument('--edge-features', default=True, action='store_true')
    parser.add_argument('--aggregators', nargs='*', default=['hadamard'])
    parser.add_argument('--classifier', default='rf', choices=['mlp', 'lr', 'svm', 'mlp_2', 'rf'])
    parser.add_argument('--coexp-thrs', nargs='*', type=str, default=['0.59', '0.75'])
    parser.add_argument('--save-predictions', default=True, action='store_true')
    parser.add_argument('--emb-size', type=int, default=16)
    parser.add_argument('--force', default=False, action='store_true')
    parser.add_argument('--inter-ratio', type=float, default=0.3)
    parser.add_argument('--test', default=False, action='store_true')

    # Node2vec params
    parser.add_argument('--num-walks', type=int, default=10)
    parser.add_argument('--walk-len', type=int, default=80)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--q', type=float, default=1.0)
    parser.add_argument('--window', type=int, default=10)

    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--n-jobs', type=int, default=10)

    args = parser.parse_args()

    seed = 42
    np.random.seed(seed)

    set_gpu(args.gpu)
    set_n_threads(args.n_jobs)

    main(args)
