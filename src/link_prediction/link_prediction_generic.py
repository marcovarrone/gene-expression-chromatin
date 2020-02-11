import argparse
import hashlib
import pickle
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import wandb

from sklearn.model_selection import train_test_split

from utils_link_prediction import *
from utils import set_n_threads, set_gpu, intra_mask

def main(args):
    args.embedding = ''
    if args.method != 'distance':
        args.embedding = 'es{}'.format(args.emb_size)

    if args.method == 'node2vec':
        args.embedding += '_nw{}_wl{}_p{}_q{}'.format(args.num_walks, args.walk_len, args.p, args.q)

    if args.chr_tgt is None:
        args.chr_tgt = args.chr_src

    if args.full_coexpression or args.full_interactions:
        chrs_coexp = 'all'
    else:
        chrs_coexp = '{}_{}'.format(args.chr_src, args.chr_tgt)

    if args.full_interactions:
        chrs_interactions = 'all'
    else:
        chrs_interactions = '{}_{}'.format(args.chr_src, args.chr_tgt)

    # ToDo: add constraint that windows and thresholds have to have the same length
    hic_files = ['primary_{}_{}'.format(type, norm) for
                 type, norm in zip(args.types, args.norms)]

    hic_preprocessings = ['{}_{}{}'.format(window, threshold, ('_' + str(weight)) if weight else '') for
                          window, threshold, weight in
                          zip(args.windows, args.hic_thrs, args.weights)]

    args.interactions = ['{}_{}_{}'.format(file, chrs_interactions, preprocessing) for
                         file, preprocessing in zip(hic_files, hic_preprocessings)]

    interactions_no_chr = ['{}_{}'.format(file, preprocessing) for
                           file, preprocessing in zip(hic_files, hic_preprocessings)]
    args.interactions = '_'.join(args.interactions)

    args.aggregators = '_'.join(args.aggregators)
    coexp_thrs = '_'.join(args.coexp_thrs)

    if args.coexp_features:
        args.folder = 'coexpression_networks'
        args.name = 'chr_{}_{}'.format(chrs_coexp, coexp_thrs)
        experiment_id = '{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, args.classifier, args.cv_splits, coexp_thrs,
                                                      args.method, args.embedding, args.aggregators)
    else:
        args.folder = 'chromatin_networks'
        args.name = args.interactions
        experiment_id = '{}_{}_{}_{}_{}_{}_'.format(args.dataset, args.classifier, args.cv_splits, coexp_thrs,
                                                    args.method, '_'.join(interactions_no_chr), args.embedding)
        experiment_id += '_' + str(args.aggregators)

    args.embedding = args.name + '_' + args.embedding

    id_hash = str(int(hashlib.sha1(experiment_id.encode()).hexdigest(), 16) % (10 ** 8))

    os.makedirs('../../results/{}/chr_{}'.format(args.dataset, args.chr_src), exist_ok=True)
    os.makedirs('../../results/{}/predictions/chr_{}'.format(args.dataset, args.chr_src), exist_ok=True)

    if args.method == 'topological':
        if args.full_coexpression:
            filename = '{}chr_all/{}_{}_{}_{}.pkl'.format('test/' if args.test else '', args.classifier,
                                                          args.method, args.name, args.aggregators)
        else:
            filename = '{}chr_{}/{}_{}_{}_{}.pkl'.format('test/' if args.test else '', args.chr_src, args.classifier,
                                                         args.method, args.name, args.aggregators)
    else:
        if args.full_coexpression:
            filename = '{}chr_all/{}_{}_{}_{}_{}.pkl'.format('test/' if args.test else '', args.classifier, args.method,
                                                             args.embedding, args.aggregators, coexp_thrs)
        else:
            filename = '{}chr_{}/{}_{}_{}_{}_{}.pkl'.format('test/' if args.test else '', args.chr_src, args.classifier,
                                                            args.method, args.embedding, args.aggregators, coexp_thrs)

    if not os.path.exists('../../results/{}/{}'.format(args.dataset, filename)) or args.force:
        coexpression = np.load(
            '../../data/{}/coexpression_networks/coexpression_chr_{}_{}.npy'.format(args.dataset, chrs_coexp,
                                                                                    coexp_thrs))
        chr_sizes = np.load(
            '../../data/{}/chr_sizes.npy'.format(args.dataset))

        disconnected_nodes = np.load(
            '../../data/{}/disconnected_nodes/{}.npy'.format(
                args.dataset, args.name))

        start_src = None
        end_src = None

        if args.full_interactions and not args.full_coexpression:
            start_src = int(np.sum(chr_sizes[:args.chr_src]))
            end_src = int(start_src + chr_sizes[args.chr_src])

            start_tgt = int(np.sum(chr_sizes[:args.chr_tgt]))
            end_tgt = int(start_tgt + chr_sizes[args.chr_tgt])

            coexpression = coexpression[start_src:end_src, start_tgt:end_tgt]

            disconnected_nodes_src = disconnected_nodes[
                                         (disconnected_nodes >= start_src) & (disconnected_nodes < end_src)] - start_src
            disconnected_nodes_tgt = disconnected_nodes[
                                         (disconnected_nodes >= start_tgt) & (disconnected_nodes < end_tgt)] - start_tgt
        else:
            disconnected_nodes_src = disconnected_nodes
            disconnected_nodes_tgt = disconnected_nodes

        print("N. disconnected nodes:", len(disconnected_nodes_src))
        if len(disconnected_nodes) > 0:
            coexpression[disconnected_nodes_src] = 0
            coexpression[:, disconnected_nodes_tgt] = 0

        n_nodes = coexpression.shape[0]

        if args.full_coexpression:
            shapes = [np.load(
                '../../data/{}/coexpression/coexpression_chr_{}_{}.npy'.format(args.dataset, i, i)).shape for i
                      in
                      range(1, 23)]

            mask = intra_mask(shapes)

            coexpression_intra = coexpression * mask
        else:
            coexpression_intra = coexpression
            mask = None

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

        if not args.coexp_intra:
            coexpression_inter = coexpression * np.logical_not(mask)

            edges_inter = np.array(np.argwhere(coexpression_inter == 1))

            if edges_intra.shape[0] > edges_inter.shape[0]:
                n_edges_inter = edges_inter.shape[0]
            else:
                n_edges_inter = int(edges_intra.shape[0] * args.inter_ratio)
            print('N. intra edges', edges_intra.shape[0], '- N. inter edges ', edges_inter.shape[0], '->',
                  n_edges_inter)
            edges_inter = edges_inter[
                np.random.choice(edges_inter.shape[0], n_edges_inter, replace=False)]
            edges_inter_nodes = np.unique(edges_inter)

            non_nodes_inter = np.setdiff1d(np.arange(n_nodes), edges_inter_nodes)

            coexpression_inter_neg = coexpression_inter.copy()
            coexpression_inter_neg[non_nodes_inter, :] = np.nan
            coexpression_inter_neg[:, non_nodes_inter] = np.nan

            non_edges_inter = np.array(np.argwhere(coexpression_inter_neg == 0))
            non_edges_inter = non_edges_inter[
                np.random.choice(non_edges_inter.shape[0], edges_inter.shape[0], replace=False)]

            edges = np.vstack((edges, edges_inter))
            non_edges = np.vstack((non_edges, non_edges_inter))

        n_edges = edges.shape[0]
        n_non_edges = non_edges.shape[0]

        if args.wandb:
            wandb.init(project="coexp-inference-models")
            wandb.config.update({'id': id_hash,
                                 'dataset': args.dataset,
                                 'fold': args.cv_splits,
                                 'windows': '_'.join(map(str, args.windows)),
                                 'chr src': args.chr_src,
                                 'chr tgt': args.chr_tgt,
                                 'hic thresholds': '_'.join(map(str, args.hic_thrs)),
                                 'coexp threshold': coexp_thrs,
                                 'full interactions': args.full_interactions,
                                 'full coexpression': args.full_coexpression,
                                 'embedding method': args.method,
                                 'aggregators': args.aggregators,
                                 'classifier': args.classifier,
                                 'interactions': args.interactions,
                                 'embeddings size': args.emb_size,
                                 'test': args.test})

        if args.method == 'topological':
            X = topological_features(args, edges, non_edges)
        elif args.method == 'ids':
            X = np.vstack((edges, non_edges))
        elif args.method == 'distance':
            X = distance_embedding(args.full_interactions, args.dataset, args.chr_src, edges, non_edges)
        else:
            X = method_embedding(args, n_nodes, edges, non_edges, start_src, end_src)
        y = np.hstack((np.ones(n_edges), np.zeros(n_non_edges)))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

        results = defaultdict(list)
        if args.test:
            results = evaluate_embedding(X_train, y_train, args.classifier, verbose=1, clf_params={'n_estimators': 100},
                                         mask=mask, X_test=X_test, y_test=y_test)
            if args.wandb:
                wandb.run.summary.update({'accuracy': results['acc'],
                                          'accuracy std': results['acc'],
                                          'precision': results['precision'],
                                          'precision std': results['precision'],
                                          'recall': results['recall'],
                                          'recall std': results['recall']})
        else:
            for i in range(args.n_iter):
                results_iter = evaluate_embedding(X_train, y_train, args.classifier, verbose=1, clf_params={'n_estimators': 100}, cv_splits=args.cv_splits, mask=mask)
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
    else:
        print('Result already computed for {}. Skipped.'.format(filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='prostate')
    parser.add_argument('--chr-src', type=int, default=2)
    parser.add_argument('--chr-tgt', type=int, default=None)
    parser.add_argument('--n-iter', type=int, default=1)
    parser.add_argument('--cv-splits', type=int, default=5)
    parser.add_argument('--method', type=str, default='topological',
                        choices=['random', 'distance', 'topological', 'svd', 'node2vec'])

    parser.add_argument('--windows', type=int, nargs='*', default=[40000])
    parser.add_argument('--hic-thrs', type=str, nargs='*', default=['3.59'])
    parser.add_argument('--weights', type=str, nargs='*', default=[None])
    parser.add_argument('--norms', nargs='*', type=str, choices=['NONE', 'VC', 'VC_SQRT', 'KR', 'ICE'],
                        default=['ICE'])
    parser.add_argument('--types', nargs='*', type=str, choices=['observed'], default=['observed', 'observed'])

    parser.add_argument('--coexp-features', default=False, action='store_true')
    parser.add_argument('--edge-features', default=True, action='store_true')
    parser.add_argument('--aggregators', nargs='*', default=['avg', 'sub'])
    parser.add_argument('--classifier', default='rf', choices=['mlp', 'lr', 'svm', 'mlp_2', 'rf'])
    parser.add_argument('--full-interactions', default=True, action='store_true')
    parser.add_argument('--full-coexpression', default=True, action='store_true')
    parser.add_argument('--coexp-thrs', nargs='*', type=str, default=['0.65'])
    parser.add_argument('--save-predictions', default=True, action='store_true')
    parser.add_argument('--emb-size', type=int, default=16)
    parser.add_argument('--force', default=False, action='store_true')
    parser.add_argument('--coexp-intra', default=True, action='store_true')
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
