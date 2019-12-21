import argparse
import os
import pickle

import networkx as nx
import numpy as np
from bionev.utils import load_embedding
from sklearn.model_selection import train_test_split

from link_prediction.utils import set_gpu, set_n_threads, evaluate_embedding

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='GM19238')
    parser.add_argument('--interactions', type=str, required=True)
    parser.add_argument('--emb-size', type=int, default=8)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--n-jobs', type=int, default=10)
    parser.add_argument('--all-regions', default=False, action='store_true')
    parser.add_argument('--chr', type=int, default=11)
    parser.add_argument('--method', default='gae', choices=['gae', 'gvae'])
    parser.add_argument('--classifier', default='mlp', choices=['mlp', 'lr', 'svm', 'mlp_2'])
    parser.add_argument('--n-runs', type=int, default=10)
    parser.add_argument('--weighted', default='False', type=str)
    parser.add_argument('--threshold', type=int, default=90)
    args = parser.parse_args()

    seed = 42
    np.random.seed(seed)

    set_gpu(args.gpu)
    set_n_threads(args.n_jobs)

    interactions_path = './data/{}/interactions/interactions_{}.npy'.format(
        args.dataset, args.interactions)

    emb_path = '{}_es{}_h{}_e{}_lr{}_do{}'.format(args.interactions, args.emb_size, args.hidden, args.epochs,
                                                  args.lr, args.dropout)

    graph = nx.from_numpy_array(np.load(interactions_path))

    if args.weighted == 'True':
        nx.write_weighted_edgelist(graph, 'data/{}/interactions/{}.edgelist'.format(args.dataset, args.interactions))
    else:
        nx.write_edgelist(graph, 'data/{}/interactions/{}.edgelist'.format(args.dataset, args.interactions))

    coexpression = np.load(
        'data/{}/coexpression/coexpression_chr_{:02d}_{}.npy'.format(args.dataset, args.chr, args.threshold))
    graph_coexp = nx.from_numpy_array(coexpression)

    edges = np.array(list(graph_coexp.edges))
    n_edges = edges.shape[0]

    non_edges = np.array(list(nx.non_edges(graph_coexp)))
    non_edges = non_edges[np.random.choice(non_edges.shape[0], n_edges, replace=False)]
    n_non_edges = non_edges.shape[0]

    accuracies = []
    f1s = []

    for run in range(args.n_runs):
        command = 'bionev --input data/{}/interactions/{}.edgelist '.format(args.dataset, args.interactions) + \
                  '--output ./embeddings/{}/{}_stability.txt '.format(args.method, emb_path) + \
                  '--method GAE --task link-prediction ' + \
                  '--gae_model_selection {} '.format('gcn_ae' if args.method == 'gae' else 'gcn_vae') + \
                  '--dimensions {} '.format(args.emb_size) + \
                  '--epochs {} '.format(args.epochs) + \
                  '--hidden {} '.format(args.hidden) + \
                  '--lr {} '.format(args.lr) + \
                  '--dropout {} '.format(args.dropout) + \
                  '--weighted {}'.format(args.weighted)
        print(command)
        os.system(command)

        emb_dict = load_embedding('./embeddings/{}/{}_stability.txt'.format(args.method, emb_path))
        emb = np.zeros((len(emb_dict.keys()), args.emb_size))
        for i in range(emb.shape[0]):
            emb[i, :] = emb_dict[str(i)]

        pos_features = np.array(list(map(lambda edge: emb[edge[0]] * emb[edge[1]], edges)))
        neg_features = np.array(list(map(lambda edge: emb[edge[0]] * emb[edge[1]], non_edges)))
        X = np.vstack((pos_features, neg_features))
        y = np.hstack((np.ones(n_edges), np.zeros(n_non_edges)))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        results = evaluate_embedding(X_train, y_train, args.classifier, n_iter=2, seed=seed)
        accuracies.append(np.mean(results['acc']))
        f1s.append(np.mean(results['f1']))

    print('Accuracies', accuracies)
    print('F1s', f1s)
    print('Accuracy mean:', np.mean(accuracies), 'std:', np.std(accuracies), '- F1 mean:', np.mean(f1s), 'std:',
          np.std(f1s))
    results = {'acc': accuracies, 'f1': f1s}
    with open('results/{}/stability/{}.pkl'.format(args.dataset, emb_path), 'wb') as file_save:
        pickle.dump(results, file_save)

    os.remove('./embeddings/{}/{}_stability.txt'.format(args.method, emb_path))
    os.remove('data/{}/interactions/{}.edgelist'.format(args.dataset, args.interactions))
