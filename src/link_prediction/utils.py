import os
from time import time

import networkx as nx
import numpy as np
import scipy.sparse as sps
from bionev.utils import load_embedding
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def evaluate_embedding(X_train, y_train, classifier_name, n_iter=10, seed=42, verbose=1, clf_params={}):
    rocs = []
    accuracies = []
    f1s = []
    precisions = []
    recalls = []
    for i in range(n_iter):
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        for train_index, val_index in skf.split(X_train, y_train):
            if classifier_name == 'mlp':
                classifier = MLPClassifier(max_iter=500, hidden_layer_sizes=(800,))
            elif classifier_name == 'svm':
                classifier = SVC(gamma='scale')
            elif classifier_name == 'mlp_2':
                classifier = MLPClassifier(max_iter=500, hidden_layer_sizes=(100, 100,), random_state=seed)
            elif classifier_name == 'rf':
                classifier = RandomForestClassifier(n_jobs=10, **clf_params)
            else:
                classifier = LogisticRegression(max_iter=500, solver='lbfgs')

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train[train_index], y_train[train_index])
            X_valid_scaled = scaler.transform(X_train[val_index])

            start = time()
            classifier.fit(X_train_scaled, y_train[train_index])
            y_pred = classifier.predict(X_valid_scaled)
            end = time()
            print('Running time:', end - start)

            roc = roc_auc_score(y_train[val_index], y_pred)
            accuracy = classifier.score(X_valid_scaled, y_train[val_index])
            f1 = f1_score(y_train[val_index], y_pred)
            precision = precision_score(y_train[val_index], y_pred)
            recall = recall_score(y_train[val_index], y_pred)
            rocs.append(roc)
            accuracies.append(accuracy)
            f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)

            if verbose:
                print("Accuracy:", accuracy, "- ROC:", roc, "- F1:", f1, "- Precision:", precision, "- Recall:", recall)
    return {'acc': accuracies, 'roc': rocs, 'f1': f1s, 'precision': precisions, 'recall': recalls}


def generate_embedding(args, emb_path, interactions_path, command):
    if not os.path.exists('embeddings/{}/{}/{}.npy'.format(args.dataset, args.method, emb_path)) or args.force:
        adj = np.load(interactions_path)
        adj[np.isnan(adj)] = 0
        if adj.shape[0] != adj.shape[1]:
            # adj = np.block([[np.zeros((adj.shape[0], adj.shape[0])), adj], [adj.T, np.zeros((adj.shape[1], adj.shape[1]))]])
            graph = nx.algorithms.bipartite.from_biadjacency_matrix(sps.csr_matrix(adj))
        else:
            graph = nx.from_numpy_array(adj)

        if args.weighted == 'True':
            nx.write_weighted_edgelist(graph,
                                       'data/{}/{}/{}{}.edgelist'.format(args.dataset, args.folder, args.name,
                                                                         '_' + str(
                                                                             args.threshold) if args.threshold else ''))
        else:
            nx.write_edgelist(graph, 'data/{}/{}/{}{}.edgelist'.format(args.dataset, args.folder, args.name,
                                                                       '_' + str(
                                                                           args.threshold) if args.threshold else ''),
                              data=False)

        print(command)
        os.system(command)
        emb_dict = load_embedding(
            '/home/varrone/Prj/gene-expression-chromatin/src/link_prediction/embeddings/{}/{}/{}.txt'.format(
                args.dataset, args.method.lower(), emb_path))

        emb = np.zeros((adj.shape[0], args.emb_size))

        disconnected_nodes = []

        print('N. genes', adj.shape[0])
        for gene in range(adj.shape[0]):
            try:
                emb[gene, :] = emb_dict[str(gene)]
            except KeyError:
                print('KeyError for', gene)
                # np.delete(emb, i, axis=0)
                emb[gene, :] = np.zeros(args.emb_size)
                disconnected_nodes.append(gene)

        np.save(
            '/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/disconnected_nodes/{}.npy'.format(
                args.dataset, args.name), np.array(disconnected_nodes))

        if args.save_emb:
            np.save('./embeddings/{}/{}/{}.npy'.format(args.dataset, args.method.lower(), emb_path), emb)
        os.remove('./embeddings/{}/{}/{}.txt'.format(args.dataset, args.method.lower(), emb_path))
        os.remove('data/{}/interactions/{}.edgelist'.format(args.dataset, args.interactions))
        return emb


def set_gpu(active=False):
    if not active:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def set_n_threads(n_threads):
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
